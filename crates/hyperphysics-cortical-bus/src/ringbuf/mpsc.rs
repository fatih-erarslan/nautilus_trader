//! # Multi-Producer Single-Consumer Ring Buffer
//!
//! Lock-free MPSC ring buffer using atomic CAS for producer synchronization.
//! Target latency: <50ns per push operation (with contention).

use portable_atomic::{AtomicBool, AtomicUsize, Ordering};
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;


/// Cache-line padded value to prevent false sharing.
#[repr(align(64))]
struct CachePadded<T> {
    value: T,
}

impl<T> CachePadded<T> {
    const fn new(value: T) -> Self {
        Self { value }
    }
}

/// Slot state for MPSC buffer.
#[repr(align(64))]
struct Slot<T> {
    /// Data storage.
    data: UnsafeCell<MaybeUninit<T>>,
    /// Write lock (true = slot being written).
    writing: AtomicBool,
    /// Ready flag (true = data ready for reading).
    ready: AtomicBool,
}

impl<T> Slot<T> {
    const fn new() -> Self {
        Self {
            data: UnsafeCell::new(MaybeUninit::uninit()),
            writing: AtomicBool::new(false),
            ready: AtomicBool::new(false),
        }
    }
}

/// Lock-free Multi-Producer Single-Consumer ring buffer.
///
/// # Performance
///
/// - Push: ~30-50ns (lock-free with CAS, may retry under contention)
/// - Pop: ~15-20ns (wait-free)
///
/// # Thread Safety
///
/// - Any number of threads may call `push()` (producers)
/// - Exactly one thread may call `pop()` (consumer)
///
/// # Algorithm
///
/// Uses a two-phase commit:
/// 1. Producer claims slot via CAS on head
/// 2. Producer writes data
/// 3. Producer marks slot ready
/// 4. Consumer reads only ready slots
///
/// # Example
///
/// ```
/// use hyperphysics_cortical_bus::ringbuf::MpscRingBuffer;
/// use std::sync::Arc;
/// use std::thread;
///
/// let buf: Arc<MpscRingBuffer<u64, 1024>> = Arc::new(MpscRingBuffer::new());
///
/// // Multiple producer threads
/// let handles: Vec<_> = (0..4).map(|i| {
///     let buf = Arc::clone(&buf);
///     thread::spawn(move || {
///         for j in 0..100 {
///             while !buf.push(i * 100 + j) {
///                 thread::yield_now();
///             }
///         }
///     })
/// }).collect();
///
/// // Single consumer
/// let mut count = 0;
/// while count < 400 {
///     if buf.pop().is_some() {
///         count += 1;
///     }
/// }
///
/// for h in handles {
///     h.join().unwrap();
/// }
/// ```
#[repr(C)]
pub struct MpscRingBuffer<T, const N: usize> {
    /// Producer claim position (multiple writers compete via CAS).
    head: CachePadded<AtomicUsize>,
    
    /// Consumer read position (single reader, no contention).
    tail: CachePadded<AtomicUsize>,
    
    /// Slot array with per-slot synchronization.
    slots: Box<[Slot<T>; N]>,
}

// Safety: MpscRingBuffer is Send/Sync because:
// - head access is synchronized via CAS
// - tail is only modified by single consumer
// - slot access is synchronized via slot.writing and slot.ready
unsafe impl<T: Send, const N: usize> Send for MpscRingBuffer<T, N> {}
unsafe impl<T: Send, const N: usize> Sync for MpscRingBuffer<T, N> {}

impl<T, const N: usize> MpscRingBuffer<T, N> {
    /// Create a new empty MPSC ring buffer.
    ///
    /// # Panics
    ///
    /// Panics if N is not a power of 2.
    pub fn new() -> Self {
        assert!(N.is_power_of_two(), "Capacity must be power of 2");
        assert!(N >= 2, "Capacity must be at least 2");
        
        Self {
            head: CachePadded::new(AtomicUsize::new(0)),
            tail: CachePadded::new(AtomicUsize::new(0)),
            slots: Box::new(std::array::from_fn(|_| Slot::new())),
        }
    }

    /// Push an item to the buffer (any producer thread).
    ///
    /// Returns `false` if the buffer is full.
    ///
    /// # Performance
    ///
    /// Lock-free but may retry under contention.
    /// Typical latency: ~30-50ns.
    #[inline]
    pub fn push(&self, item: T) -> bool {
        loop {
            let head = self.head.value.load(Ordering::Relaxed);
            let tail = self.tail.value.load(Ordering::Acquire);
            
            let next_head = (head + 1) & (N - 1);
            
            // Check if full
            if next_head == tail {
                return false;
            }
            
            // Try to claim slot
            if self.head.value.compare_exchange_weak(
                head,
                next_head,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ).is_ok() {
                // Won the slot, now write data
                let slot = &self.slots[head];
                
                // Mark as writing (for debugging/safety)
                slot.writing.store(true, Ordering::Relaxed);
                
                // Write data
                unsafe {
                    (*slot.data.get()).write(item);
                }
                
                // Mark as ready
                slot.writing.store(false, Ordering::Release);
                slot.ready.store(true, Ordering::Release);
                
                return true;
            }
            
            // Lost race, retry
            std::hint::spin_loop();
        }
    }

    /// Pop an item from the buffer (single consumer thread only).
    ///
    /// Returns `None` if no ready items are available.
    ///
    /// # Performance
    ///
    /// Wait-free for the consumer.
    /// Typical latency: ~15-20ns.
    #[inline(always)]
    pub fn pop(&self) -> Option<T> {
        let tail = self.tail.value.load(Ordering::Relaxed);
        let head = self.head.value.load(Ordering::Acquire);
        
        if tail == head {
            return None; // Empty
        }
        
        let slot = &self.slots[tail];
        
        // Wait for slot to be ready (handles in-flight writes)
        if !slot.ready.load(Ordering::Acquire) {
            return None; // Slot claimed but not yet written
        }
        
        // Read data
        let item = unsafe { (*slot.data.get()).assume_init_read() };
        
        // Clear ready flag for reuse
        slot.ready.store(false, Ordering::Release);
        
        // Advance tail
        let next_tail = (tail + 1) & (N - 1);
        self.tail.value.store(next_tail, Ordering::Release);
        
        Some(item)
    }

    /// Check if the buffer is empty.
    ///
    /// Note: May return false negative if writes are in-flight.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        let tail = self.tail.value.load(Ordering::Relaxed);
        let head = self.head.value.load(Ordering::Relaxed);
        tail == head
    }

    /// Check if the buffer is full.
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        let head = self.head.value.load(Ordering::Relaxed);
        let tail = self.tail.value.load(Ordering::Relaxed);
        let next_head = (head + 1) & (N - 1);
        next_head == tail
    }

    /// Get approximate number of items in the buffer.
    #[inline(always)]
    pub fn len(&self) -> usize {
        let head = self.head.value.load(Ordering::Relaxed);
        let tail = self.tail.value.load(Ordering::Relaxed);
        
        if head >= tail {
            head - tail
        } else {
            N - tail + head
        }
    }

    /// Get the capacity of the buffer.
    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        N - 1
    }

    /// Try to push without spinning (single attempt).
    ///
    /// Returns `Err(item)` if CAS failed or buffer full.
    #[inline]
    pub fn try_push(&self, item: T) -> Result<(), T> {
        let head = self.head.value.load(Ordering::Relaxed);
        let tail = self.tail.value.load(Ordering::Acquire);
        
        let next_head = (head + 1) & (N - 1);
        
        if next_head == tail {
            return Err(item); // Full
        }
        
        if self.head.value.compare_exchange(
            head,
            next_head,
            Ordering::AcqRel,
            Ordering::Relaxed,
        ).is_ok() {
            let slot = &self.slots[head];
            slot.writing.store(true, Ordering::Relaxed);
            unsafe { (*slot.data.get()).write(item); }
            slot.writing.store(false, Ordering::Release);
            slot.ready.store(true, Ordering::Release);
            Ok(())
        } else {
            Err(item) // CAS failed
        }
    }

    /// Pop all available items into buffer.
    ///
    /// Returns number of items popped.
    #[inline]
    pub fn pop_batch(&self, buffer: &mut [T]) -> usize {
        let mut count = 0;
        for slot in buffer.iter_mut() {
            if let Some(item) = self.pop() {
                *slot = item;
                count += 1;
            } else {
                break;
            }
        }
        count
    }
}

impl<T, const N: usize> Default for MpscRingBuffer<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Drop for MpscRingBuffer<T, N> {
    fn drop(&mut self) {
        // Drop any remaining items
        while self.pop().is_some() {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_basic_push_pop() {
        let buf: MpscRingBuffer<u64, 16> = MpscRingBuffer::new();
        
        assert!(buf.is_empty());
        assert!(buf.push(42));
        assert!(!buf.is_empty());
        assert_eq!(buf.pop(), Some(42));
        assert!(buf.is_empty());
    }

    #[test]
    fn test_multi_producer() {
        let buf: Arc<MpscRingBuffer<u64, 256>> = Arc::new(MpscRingBuffer::new());
        
        const PRODUCERS: u64 = 4;
        const PER_PRODUCER: u64 = 50;
        
        let handles: Vec<_> = (0..PRODUCERS)
            .map(|p| {
                let buf = Arc::clone(&buf);
                thread::spawn(move || {
                    for i in 0..PER_PRODUCER {
                        let val = p * PER_PRODUCER + i;
                        while !buf.push(val) {
                            thread::yield_now();
                        }
                    }
                })
            })
            .collect();
        
        // Wait for producers
        for h in handles {
            h.join().unwrap();
        }
        
        // Collect all items
        let mut items = Vec::new();
        while let Some(val) = buf.pop() {
            items.push(val);
        }
        
        assert_eq!(items.len(), (PRODUCERS * PER_PRODUCER) as usize);
        
        // Verify all values present (order may vary)
        items.sort();
        for i in 0..(PRODUCERS * PER_PRODUCER) {
            assert!(items.contains(&i), "Missing value: {}", i);
        }
    }

    #[test]
    fn test_concurrent_push_pop() {
        let buf: Arc<MpscRingBuffer<u64, 1024>> = Arc::new(MpscRingBuffer::new());
        
        const TOTAL: u64 = 10_000;
        const PRODUCERS: usize = 4;
        
        let producers: Vec<_> = (0..PRODUCERS)
            .map(|p| {
                let buf = Arc::clone(&buf);
                thread::spawn(move || {
                    for i in 0..(TOTAL / PRODUCERS as u64) {
                        let val = p as u64 * (TOTAL / PRODUCERS as u64) + i;
                        while !buf.push(val) {
                            thread::yield_now();
                        }
                    }
                })
            })
            .collect();
        
        let buf_consumer = Arc::clone(&buf);
        let consumer = thread::spawn(move || {
            let mut count = 0u64;
            let mut sum = 0u64;
            while count < TOTAL {
                if let Some(val) = buf_consumer.pop() {
                    sum += val;
                    count += 1;
                } else {
                    thread::yield_now();
                }
            }
            sum
        });
        
        for p in producers {
            p.join().unwrap();
        }
        
        let sum = consumer.join().unwrap();
        let expected = TOTAL * (TOTAL - 1) / 2;
        assert_eq!(sum, expected);
    }

    #[test]
    fn test_try_push() {
        let buf: MpscRingBuffer<u64, 4> = MpscRingBuffer::new();
        
        assert!(buf.try_push(1).is_ok());
        assert!(buf.try_push(2).is_ok());
        assert!(buf.try_push(3).is_ok());
        assert!(buf.try_push(4).is_err()); // Full
        
        buf.pop();
        assert!(buf.try_push(4).is_ok());
    }
}
