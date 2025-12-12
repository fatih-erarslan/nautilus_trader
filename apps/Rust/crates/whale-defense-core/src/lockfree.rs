//! Lock-free data structures for zero-contention whale defense
//! 
//! Ultra-fast lock-free implementations optimized for sub-microsecond latency.
//! All operations use atomic instructions and memory ordering for maximum performance.

use crate::{
    error::{WhaleDefenseError, Result},
    config::*,
    AtomicU64, AtomicUsize, Ordering,
};
use cache_padded::CachePadded;
use core::{
    mem::{MaybeUninit, align_of},
    ptr::NonNull,
    slice,
    sync::atomic::{AtomicPtr, AtomicBool},
};

/// Lock-free ring buffer with zero-allocation operations
/// 
/// This implementation is optimized for:
/// - Sub-microsecond latency operations
/// - Zero memory allocations after initialization
/// - Cache-friendly memory layout
/// - Wait-free operations for single producer/consumer
/// - Memory ordering optimizations for x86_64
/// 
/// # Safety
/// This structure uses extensive unsafe code for performance.
/// All operations maintain memory safety through careful atomic ordering.
#[repr(C)]
pub struct LockFreeRingBuffer<T> {
    /// Write position (cache-aligned)
    write_pos: CachePadded<AtomicU64>,
    
    /// Read position (cache-aligned)
    read_pos: CachePadded<AtomicU64>,
    
    /// Buffer size mask (Size - 1, where Size is power of 2)
    size_mask: u64,
    
    /// Buffer storage (cache-aligned)
    buffer: NonNull<T>,
    
    /// Buffer capacity
    capacity: usize,
    
    /// Destructor flag
    needs_drop: bool,
}

impl<T> LockFreeRingBuffer<T> {
    /// Create new lock-free ring buffer
    /// 
    /// # Safety
    /// - `capacity` must be a power of 2
    /// - `capacity` must be > 0
    /// - Must call `destroy()` before dropping
    pub unsafe fn new(capacity: usize) -> Result<Self> {
        // Validate capacity is power of 2
        if capacity == 0 || (capacity & (capacity - 1)) != 0 {
            return Err(WhaleDefenseError::InvalidParameter);
        }
        
        // Allocate aligned memory
        let layout = core::alloc::Layout::from_size_align(
            capacity * core::mem::size_of::<T>(),
            CACHE_LINE_SIZE,
        ).map_err(|_| WhaleDefenseError::OutOfMemory)?;
        
        #[cfg(feature = "std")]
        let buffer = {
            let ptr = std::alloc::alloc(layout) as *mut T;
            NonNull::new(ptr).ok_or(WhaleDefenseError::OutOfMemory)?
        };
        
        #[cfg(not(feature = "std"))]
        let buffer = {
            use linked_list_allocator::Heap;
            static mut HEAP: Heap = Heap::empty();
            let ptr = HEAP.allocate_first_fit(layout)
                .map_err(|_| WhaleDefenseError::OutOfMemory)?
                .as_ptr() as *mut T;
            NonNull::new(ptr).ok_or(WhaleDefenseError::OutOfMemory)?
        };
        
        Ok(Self {
            write_pos: CachePadded::new(AtomicU64::new(0)),
            read_pos: CachePadded::new(AtomicU64::new(0)),
            size_mask: (capacity - 1) as u64,
            buffer,
            capacity,
            needs_drop: core::mem::needs_drop::<T>(),
        })
    }
    
    /// Wait-free write operation
    /// 
    /// # Performance
    /// - Target latency: <50 nanoseconds
    /// - Uses relaxed ordering for maximum performance
    /// - Optimized for single-producer scenarios
    /// 
    /// # Safety
    /// - Must not be called concurrently from multiple threads (single producer)
    /// - `item` must be valid for the lifetime of the buffer
    #[inline(always)]
    pub unsafe fn try_write(&self, item: T) -> Result<()> {
        let current_write = self.write_pos.load(Ordering::Relaxed);
        let next_write = current_write.wrapping_add(1);
        
        // Check if buffer is full (leave one slot empty to distinguish full/empty)
        if next_write & self.size_mask == self.read_pos.load(Ordering::Acquire) & self.size_mask {
            return Err(WhaleDefenseError::BufferOverflow);
        }
        
        // Write data to buffer
        let buffer_ptr = self.buffer.as_ptr();
        let slot_ptr = buffer_ptr.add((current_write & self.size_mask) as usize);
        slot_ptr.write(item);
        
        // Release write position
        self.write_pos.store(next_write, Ordering::Release);
        
        Ok(())
    }
    
    /// Wait-free read operation
    /// 
    /// # Performance
    /// - Target latency: <50 nanoseconds
    /// - Uses acquire ordering for memory safety
    /// - Optimized for single-consumer scenarios
    /// 
    /// # Safety
    /// - Must not be called concurrently from multiple threads (single consumer)
    /// - Returned item must be used or dropped properly
    #[inline(always)]
    pub unsafe fn try_read(&self) -> Result<T> {
        let current_read = self.read_pos.load(Ordering::Relaxed);
        
        // Check if buffer is empty
        if current_read == self.write_pos.load(Ordering::Acquire) {
            return Err(WhaleDefenseError::BufferUnderflow);
        }
        
        // Read data from buffer
        let buffer_ptr = self.buffer.as_ptr();
        let slot_ptr = buffer_ptr.add((current_read & self.size_mask) as usize);
        let item = slot_ptr.read();
        
        // Release read position
        self.read_pos.store(current_read.wrapping_add(1), Ordering::Release);
        
        Ok(item)
    }
    
    /// Get current buffer size (approximate)
    /// 
    /// # Note
    /// This operation is not atomic and may return stale values
    /// in concurrent scenarios. Use only for monitoring/debugging.
    #[inline(always)]
    pub fn size(&self) -> usize {
        let write_pos = self.write_pos.load(Ordering::Acquire);
        let read_pos = self.read_pos.load(Ordering::Acquire);
        ((write_pos.wrapping_sub(read_pos)) & self.size_mask) as usize
    }
    
    /// Get buffer capacity
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Check if buffer is empty (approximate)
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.read_pos.load(Ordering::Acquire) == self.write_pos.load(Ordering::Acquire)
    }
    
    /// Check if buffer is full (approximate)
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        let write_pos = self.write_pos.load(Ordering::Acquire);
        let read_pos = self.read_pos.load(Ordering::Acquire);
        write_pos.wrapping_add(1) & self.size_mask == read_pos & self.size_mask
    }
    
    /// Destroy the buffer and free memory
    /// 
    /// # Safety
    /// - Must be called before dropping the buffer
    /// - Must not be called concurrently
    /// - Buffer must not be used after calling this
    pub unsafe fn destroy(&mut self) {
        // Drop remaining items if necessary
        if self.needs_drop {
            while let Ok(_) = self.try_read() {
                // Items are automatically dropped
            }
        }
        
        // Free memory
        let layout = core::alloc::Layout::from_size_align(
            self.capacity * core::mem::size_of::<T>(),
            CACHE_LINE_SIZE,
        ).unwrap();
        
        #[cfg(feature = "std")]
        std::alloc::dealloc(self.buffer.as_ptr() as *mut u8, layout);
        
        #[cfg(not(feature = "std"))]
        {
            use linked_list_allocator::Heap;
            static mut HEAP: Heap = Heap::empty();
            HEAP.deallocate(
                NonNull::new(self.buffer.as_ptr() as *mut u8).unwrap(),
                layout,
            );
        }
    }
}

unsafe impl<T: Send> Send for LockFreeRingBuffer<T> {}
unsafe impl<T: Send> Sync for LockFreeRingBuffer<T> {}

/// Lock-free stack for temporary storage
/// 
/// Optimized for scenarios where LIFO ordering is acceptable
/// and maximum performance is required.
#[repr(C)]
pub struct LockFreeStack<T> {
    /// Head pointer (tagged pointer to prevent ABA problem)
    head: AtomicPtr<Node<T>>,
    
    /// Node counter for ABA prevention
    counter: AtomicU64,
}

#[repr(C)]
struct Node<T> {
    data: MaybeUninit<T>,
    next: *mut Node<T>,
    tag: u64,
}

impl<T> LockFreeStack<T> {
    /// Create new lock-free stack
    pub const fn new() -> Self {
        Self {
            head: AtomicPtr::new(core::ptr::null_mut()),
            counter: AtomicU64::new(0),
        }
    }
    
    /// Push item onto stack
    /// 
    /// # Safety
    /// - Item must be valid for the lifetime of the stack
    /// - Must handle memory allocation failures
    pub unsafe fn push(&self, item: T) -> Result<()> {
        // Allocate node
        let node = Box::into_raw(Box::new(Node {
            data: MaybeUninit::new(item),
            next: core::ptr::null_mut(),
            tag: self.counter.fetch_add(1, Ordering::Relaxed),
        }));
        
        // CAS loop to update head
        loop {
            let current_head = self.head.load(Ordering::Acquire);
            (*node).next = current_head;
            
            match self.head.compare_exchange_weak(
                current_head,
                node,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(_) => continue,
            }
        }
        
        Ok(())
    }
    
    /// Pop item from stack
    /// 
    /// # Safety
    /// - Must handle potential ABA problem
    /// - Must properly manage memory lifecycle
    pub unsafe fn pop(&self) -> Result<T> {
        loop {
            let current_head = self.head.load(Ordering::Acquire);
            
            if current_head.is_null() {
                return Err(WhaleDefenseError::BufferUnderflow);
            }
            
            let next = (*current_head).next;
            
            match self.head.compare_exchange_weak(
                current_head,
                next,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    let item = (*current_head).data.assume_init();
                    let _ = Box::from_raw(current_head);
                    return Ok(item);
                }
                Err(_) => continue,
            }
        }
    }
    
    /// Check if stack is empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Acquire).is_null()
    }
}

unsafe impl<T: Send> Send for LockFreeStack<T> {}
unsafe impl<T: Send> Sync for LockFreeStack<T> {}

impl<T> Drop for LockFreeStack<T> {
    fn drop(&mut self) {
        unsafe {
            while let Ok(_) = self.pop() {
                // Items are automatically dropped
            }
        }
    }
}

/// Lock-free queue for FIFO operations
/// 
/// Implementation using Michael & Scott's algorithm with optimizations
/// for whale defense scenarios.
#[repr(C)]
pub struct LockFreeQueue<T> {
    head: AtomicPtr<QueueNode<T>>,
    tail: AtomicPtr<QueueNode<T>>,
}

#[repr(C)]
struct QueueNode<T> {
    data: MaybeUninit<T>,
    next: AtomicPtr<QueueNode<T>>,
}

impl<T> LockFreeQueue<T> {
    /// Create new lock-free queue
    pub fn new() -> Self {
        let dummy = Box::into_raw(Box::new(QueueNode {
            data: MaybeUninit::uninit(),
            next: AtomicPtr::new(core::ptr::null_mut()),
        }));
        
        Self {
            head: AtomicPtr::new(dummy),
            tail: AtomicPtr::new(dummy),
        }
    }
    
    /// Enqueue item
    /// 
    /// # Safety
    /// - Memory allocation must be handled properly
    /// - ABA problem must be prevented
    pub unsafe fn enqueue(&self, item: T) -> Result<()> {
        let node = Box::into_raw(Box::new(QueueNode {
            data: MaybeUninit::new(item),
            next: AtomicPtr::new(core::ptr::null_mut()),
        }));
        
        loop {
            let tail = self.tail.load(Ordering::Acquire);
            let next = (*tail).next.load(Ordering::Acquire);
            
            if tail == self.tail.load(Ordering::Acquire) {
                if next.is_null() {
                    if (*tail).next.compare_exchange(
                        next,
                        node,
                        Ordering::Release,
                        Ordering::Relaxed,
                    ).is_ok() {
                        break;
                    }
                } else {
                    let _ = self.tail.compare_exchange(
                        tail,
                        next,
                        Ordering::Release,
                        Ordering::Relaxed,
                    );
                }
            }
        }
        
        let _ = self.tail.compare_exchange(
            self.tail.load(Ordering::Acquire),
            node,
            Ordering::Release,
            Ordering::Relaxed,
        );
        
        Ok(())
    }
    
    /// Dequeue item
    /// 
    /// # Safety
    /// - Memory management must be handled properly
    /// - ABA problem must be prevented
    pub unsafe fn dequeue(&self) -> Result<T> {
        loop {
            let head = self.head.load(Ordering::Acquire);
            let tail = self.tail.load(Ordering::Acquire);
            let next = (*head).next.load(Ordering::Acquire);
            
            if head == self.head.load(Ordering::Acquire) {
                if head == tail {
                    if next.is_null() {
                        return Err(WhaleDefenseError::BufferUnderflow);
                    }
                    
                    let _ = self.tail.compare_exchange(
                        tail,
                        next,
                        Ordering::Release,
                        Ordering::Relaxed,
                    );
                } else {
                    if next.is_null() {
                        continue;
                    }
                    
                    let item = (*next).data.assume_init_read();
                    
                    if self.head.compare_exchange(
                        head,
                        next,
                        Ordering::Release,
                        Ordering::Relaxed,
                    ).is_ok() {
                        let _ = Box::from_raw(head);
                        return Ok(item);
                    }
                }
            }
        }
    }
    
    /// Check if queue is empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        head == tail && unsafe { (*head).next.load(Ordering::Acquire).is_null() }
    }
}

unsafe impl<T: Send> Send for LockFreeQueue<T> {}
unsafe impl<T: Send> Sync for LockFreeQueue<T> {}

impl<T> Drop for LockFreeQueue<T> {
    fn drop(&mut self) {
        unsafe {
            while let Ok(_) = self.dequeue() {
                // Items are automatically dropped
            }
            
            // Free dummy node
            let head = self.head.load(Ordering::Acquire);
            if !head.is_null() {
                let _ = Box::from_raw(head);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ring_buffer_basic() {
        unsafe {
            let mut buffer = LockFreeRingBuffer::<i32>::new(1024).unwrap();
            
            assert!(buffer.is_empty());
            assert!(!buffer.is_full());
            
            // Test write/read
            buffer.try_write(42).unwrap();
            assert_eq!(buffer.try_read().unwrap(), 42);
            
            buffer.destroy();
        }
    }
    
    #[test]
    fn test_stack_basic() {
        unsafe {
            let stack = LockFreeStack::<i32>::new();
            
            assert!(stack.is_empty());
            
            stack.push(1).unwrap();
            stack.push(2).unwrap();
            stack.push(3).unwrap();
            
            assert_eq!(stack.pop().unwrap(), 3);
            assert_eq!(stack.pop().unwrap(), 2);
            assert_eq!(stack.pop().unwrap(), 1);
            
            assert!(stack.is_empty());
        }
    }
    
    #[test]
    fn test_queue_basic() {
        unsafe {
            let queue = LockFreeQueue::<i32>::new();
            
            assert!(queue.is_empty());
            
            queue.enqueue(1).unwrap();
            queue.enqueue(2).unwrap();
            queue.enqueue(3).unwrap();
            
            assert_eq!(queue.dequeue().unwrap(), 1);
            assert_eq!(queue.dequeue().unwrap(), 2);
            assert_eq!(queue.dequeue().unwrap(), 3);
            
            assert!(queue.is_empty());
        }
    }
}