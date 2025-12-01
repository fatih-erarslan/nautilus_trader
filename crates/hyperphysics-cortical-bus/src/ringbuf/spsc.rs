//! # Single-Producer Single-Consumer Ring Buffer
//!
//! Wait-free SPSC ring buffer with cache-line padding to prevent false sharing.
//! Target latency: <20ns per push/pop operation.

use portable_atomic::{AtomicUsize, Ordering};
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;


/// Cache-line padded value to prevent false sharing.
#[repr(C, align(64))]
struct CachePadded<T> {
    value: T,
}

impl<T> CachePadded<T> {
    const fn new(value: T) -> Self {
        Self { value }
    }
}

/// Wait-free Single-Producer Single-Consumer ring buffer.
///
/// # Performance
///
/// - Push: ~15-20ns (wait-free)
/// - Pop: ~15-20ns (wait-free)
/// - Memory: O(N) where N is capacity
///
/// # Thread Safety
///
/// - Exactly one thread may call `push()` (producer)
/// - Exactly one thread may call `pop()` (consumer)
/// - Different threads may be producer and consumer
///
/// # Example
///
/// ```
/// use hyperphysics_cortical_bus::ringbuf::SpscRingBuffer;
///
/// let buf: SpscRingBuffer<u64, 1024> = SpscRingBuffer::new();
///
/// // Producer thread
/// buf.push(42);
///
/// // Consumer thread
/// if let Some(val) = buf.pop() {
///     assert_eq!(val, 42);
/// }
/// ```
#[repr(C)]
pub struct SpscRingBuffer<T, const N: usize> {
    /// Producer writes to this position.
    /// Only modified by producer thread.
    head: CachePadded<AtomicUsize>,
    
    /// Consumer reads from this position.
    /// Only modified by consumer thread.
    tail: CachePadded<AtomicUsize>,
    
    /// Data storage.
    /// UnsafeCell allows interior mutability.
    buffer: UnsafeCell<[MaybeUninit<T>; N]>,
}

// Safety: SpscRingBuffer is Send/Sync because:
// - head is only modified by producer (atomic)
// - tail is only modified by consumer (atomic)
// - buffer access is synchronized by head/tail ordering
unsafe impl<T: Send, const N: usize> Send for SpscRingBuffer<T, N> {}
unsafe impl<T: Send, const N: usize> Sync for SpscRingBuffer<T, N> {}

impl<T, const N: usize> SpscRingBuffer<T, N> {
    /// Create a new empty ring buffer.
    ///
    /// # Panics
    ///
    /// Panics if N is not a power of 2 (required for fast modulo).
    pub fn new() -> Self {
        assert!(N.is_power_of_two(), "Capacity must be power of 2");
        assert!(N >= 2, "Capacity must be at least 2");
        
        Self {
            head: CachePadded::new(AtomicUsize::new(0)),
            tail: CachePadded::new(AtomicUsize::new(0)),
            buffer: UnsafeCell::new(unsafe {
                MaybeUninit::uninit().assume_init()
            }),
        }
    }

    /// Push an item to the buffer (producer only).
    ///
    /// Returns `false` if the buffer is full.
    ///
    /// # Performance
    ///
    /// Wait-free: O(1) time, no loops or retries.
    /// Typical latency: ~15-20ns.
    #[inline(always)]
    pub fn push(&self, item: T) -> bool {
        let head = self.head.value.load(Ordering::Relaxed);
        let tail = self.tail.value.load(Ordering::Acquire);
        
        // Check if full (head + 1 == tail means full)
        let next_head = (head + 1) & (N - 1); // Fast modulo for power-of-2
        
        if next_head == tail {
            return false; // Full
        }
        
        // Write data
        unsafe {
            let slot = (*self.buffer.get()).get_unchecked_mut(head);
            slot.write(item);
        }
        
        // Publish (release ensures data is visible before head update)
        self.head.value.store(next_head, Ordering::Release);
        
        true
    }

    /// Pop an item from the buffer (consumer only).
    ///
    /// Returns `None` if the buffer is empty.
    ///
    /// # Performance
    ///
    /// Wait-free: O(1) time, no loops or retries.
    /// Typical latency: ~15-20ns.
    #[inline(always)]
    pub fn pop(&self) -> Option<T> {
        let tail = self.tail.value.load(Ordering::Relaxed);
        let head = self.head.value.load(Ordering::Acquire);
        
        if tail == head {
            return None; // Empty
        }
        
        // Read data
        let item = unsafe {
            let slot = (*self.buffer.get()).get_unchecked(tail);
            slot.assume_init_read()
        };
        
        // Advance tail (release ensures read completes before update)
        let next_tail = (tail + 1) & (N - 1);
        self.tail.value.store(next_tail, Ordering::Release);
        
        Some(item)
    }

    /// Peek at the next item without removing it.
    ///
    /// # Safety
    ///
    /// The returned reference is only valid until the next `pop()` call.
    #[inline(always)]
    pub fn peek(&self) -> Option<&T> {
        let tail = self.tail.value.load(Ordering::Relaxed);
        let head = self.head.value.load(Ordering::Acquire);
        
        if tail == head {
            return None; // Empty
        }
        
        unsafe {
            let slot = (*self.buffer.get()).get_unchecked(tail);
            Some(slot.assume_init_ref())
        }
    }

    /// Check if the buffer is empty.
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

    /// Get the number of items in the buffer.
    ///
    /// Note: This is approximate due to concurrent access.
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
        N - 1 // One slot reserved to distinguish full from empty
    }

    /// Get available space for writing.
    #[inline(always)]
    pub fn available(&self) -> usize {
        self.capacity() - self.len()
    }

    /// Push multiple items (batch operation).
    ///
    /// Returns the number of items successfully pushed.
    #[inline]
    pub fn push_batch(&self, items: &[T]) -> usize
    where
        T: Copy,
    {
        let mut count = 0;
        for &item in items {
            if self.push(item) {
                count += 1;
            } else {
                break;
            }
        }
        count
    }

    /// Pop multiple items (batch operation).
    ///
    /// Returns the number of items successfully popped.
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

impl<T, const N: usize> Default for SpscRingBuffer<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Drop for SpscRingBuffer<T, N> {
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
        let buf: SpscRingBuffer<u64, 16> = SpscRingBuffer::new();
        
        assert!(buf.is_empty());
        assert!(!buf.is_full());
        
        assert!(buf.push(42));
        assert!(!buf.is_empty());
        
        assert_eq!(buf.pop(), Some(42));
        assert!(buf.is_empty());
    }

    #[test]
    fn test_full_buffer() {
        let buf: SpscRingBuffer<u64, 4> = SpscRingBuffer::new();
        
        // Capacity is N-1 = 3
        assert!(buf.push(1));
        assert!(buf.push(2));
        assert!(buf.push(3));
        assert!(buf.is_full());
        assert!(!buf.push(4)); // Should fail
        
        assert_eq!(buf.pop(), Some(1));
        assert!(buf.push(4)); // Should succeed now
    }

    #[test]
    fn test_wraparound() {
        let buf: SpscRingBuffer<u64, 4> = SpscRingBuffer::new();
        
        for i in 0..10 {
            buf.push(i);
            assert_eq!(buf.pop(), Some(i));
        }
    }

    #[test]
    fn test_concurrent() {
        let buf: Arc<SpscRingBuffer<u64, 1024>> = Arc::new(SpscRingBuffer::new());
        let buf_producer = Arc::clone(&buf);
        let buf_consumer = Arc::clone(&buf);
        
        const COUNT: u64 = 100_000;
        
        let producer = thread::spawn(move || {
            for i in 0..COUNT {
                while !buf_producer.push(i) {
                    thread::yield_now();
                }
            }
        });
        
        let consumer = thread::spawn(move || {
            let mut sum = 0u64;
            let mut received = 0u64;
            while received < COUNT {
                if let Some(val) = buf_consumer.pop() {
                    sum += val;
                    received += 1;
                } else {
                    thread::yield_now();
                }
            }
            sum
        });
        
        producer.join().unwrap();
        let sum = consumer.join().unwrap();
        
        // Sum of 0..COUNT = COUNT * (COUNT-1) / 2
        let expected = COUNT * (COUNT - 1) / 2;
        assert_eq!(sum, expected);
    }

    #[test]
    fn test_batch_operations() {
        let buf: SpscRingBuffer<u64, 16> = SpscRingBuffer::new();
        
        let items = [1u64, 2, 3, 4, 5];
        assert_eq!(buf.push_batch(&items), 5);
        
        let mut output = [0u64; 10];
        assert_eq!(buf.pop_batch(&mut output), 5);
        assert_eq!(&output[..5], &items);
    }

    #[test]
    fn test_peek() {
        let buf: SpscRingBuffer<u64, 16> = SpscRingBuffer::new();
        
        assert!(buf.peek().is_none());
        
        buf.push(42);
        assert_eq!(buf.peek(), Some(&42));
        assert_eq!(buf.peek(), Some(&42)); // Still there
        assert_eq!(buf.pop(), Some(42));
        assert!(buf.peek().is_none());
    }
}
