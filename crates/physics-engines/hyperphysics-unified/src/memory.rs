//! Memory-efficient object pooling and arena allocation

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// Generational index for safe handle reuse
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenIndex {
    index: u32,
    generation: u32,
}

impl GenIndex {
    /// Create new generational index
    pub fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// Get raw index
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Get generation
    pub fn generation(&self) -> u32 {
        self.generation
    }
}

/// Object pool with generational indices
pub struct ObjectPool<T> {
    data: Vec<UnsafeCell<MaybeUninit<T>>>,
    generations: Vec<AtomicU32>,
    free_list: Vec<u32>,
    len: AtomicUsize,
    capacity: usize,
}

impl<T> ObjectPool<T> {
    /// Create pool with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        let mut data = Vec::with_capacity(capacity);
        let mut generations = Vec::with_capacity(capacity);

        for _ in 0..capacity {
            data.push(UnsafeCell::new(MaybeUninit::uninit()));
            generations.push(AtomicU32::new(0));
        }

        Self {
            data,
            generations,
            free_list: (0..capacity as u32).rev().collect(),
            len: AtomicUsize::new(0),
            capacity,
        }
    }

    /// Allocate object, returns generational index
    pub fn alloc(&mut self, value: T) -> Option<GenIndex> {
        let index = self.free_list.pop()?;
        let gen = self.generations[index as usize].load(Ordering::Acquire);

        // Safety: We have exclusive access via &mut self
        unsafe {
            (*self.data[index as usize].get()).write(value);
        }

        self.len.fetch_add(1, Ordering::Release);
        Some(GenIndex::new(index, gen))
    }

    /// Free object by generational index
    pub fn free(&mut self, idx: GenIndex) -> Option<T> {
        let index = idx.index() as usize;
        if index >= self.capacity {
            return None;
        }

        let current_gen = self.generations[index].load(Ordering::Acquire);
        if current_gen != idx.generation() {
            return None; // Stale handle
        }

        // Increment generation to invalidate existing handles
        self.generations[index].fetch_add(1, Ordering::Release);

        // Safety: We verified generation matches
        let value = unsafe {
            std::ptr::read((*self.data[index].get()).as_ptr())
        };

        self.free_list.push(idx.index());
        self.len.fetch_sub(1, Ordering::Release);
        Some(value)
    }

    /// Get reference by generational index
    pub fn get(&self, idx: GenIndex) -> Option<&T> {
        let index = idx.index() as usize;
        if index >= self.capacity {
            return None;
        }

        let current_gen = self.generations[index].load(Ordering::Acquire);
        if current_gen != idx.generation() {
            return None;
        }

        // Safety: Generation check passed, data is initialized
        unsafe { Some((*self.data[index].get()).assume_init_ref()) }
    }

    /// Get mutable reference by generational index
    pub fn get_mut(&mut self, idx: GenIndex) -> Option<&mut T> {
        let index = idx.index() as usize;
        if index >= self.capacity {
            return None;
        }

        let current_gen = self.generations[index].load(Ordering::Acquire);
        if current_gen != idx.generation() {
            return None;
        }

        // Safety: We have &mut self, generation check passed
        unsafe { Some((*self.data[index].get()).assume_init_mut()) }
    }

    /// Number of allocated objects
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Pool capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// Safety: T: Send implies ObjectPool<T>: Send
unsafe impl<T: Send> Send for ObjectPool<T> {}
// Safety: We use atomic operations for generation checks
unsafe impl<T: Send + Sync> Sync for ObjectPool<T> {}

/// Simple bump allocator arena
pub struct Arena {
    chunks: Vec<Vec<u8>>,
    current: usize,
    offset: usize,
    chunk_size: usize,
}

impl Arena {
    /// Create arena with chunk size
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunks: vec![vec![0u8; chunk_size]],
            current: 0,
            offset: 0,
            chunk_size,
        }
    }

    /// Allocate bytes with alignment
    pub fn alloc_bytes(&mut self, size: usize, align: usize) -> *mut u8 {
        // Align offset
        let aligned = (self.offset + align - 1) & !(align - 1);

        if aligned + size > self.chunk_size {
            // Need new chunk
            let new_size = self.chunk_size.max(size + align);
            self.chunks.push(vec![0u8; new_size]);
            self.current = self.chunks.len() - 1;
            self.offset = 0;
            return self.alloc_bytes(size, align);
        }

        self.offset = aligned + size;
        unsafe { self.chunks[self.current].as_mut_ptr().add(aligned) }
    }

    /// Allocate typed value
    pub fn alloc<T>(&mut self, value: T) -> &mut T {
        let ptr = self.alloc_bytes(
            std::mem::size_of::<T>(),
            std::mem::align_of::<T>(),
        ) as *mut T;

        unsafe {
            ptr.write(value);
            &mut *ptr
        }
    }

    /// Allocate slice
    pub fn alloc_slice<T: Clone>(&mut self, slice: &[T]) -> &mut [T] {
        let ptr = self.alloc_bytes(
            std::mem::size_of::<T>() * slice.len(),
            std::mem::align_of::<T>(),
        ) as *mut T;

        unsafe {
            for (i, item) in slice.iter().enumerate() {
                ptr.add(i).write(item.clone());
            }
            std::slice::from_raw_parts_mut(ptr, slice.len())
        }
    }

    /// Reset arena (invalidates all allocations!)
    pub fn reset(&mut self) {
        self.current = 0;
        self.offset = 0;
    }

    /// Total allocated bytes
    pub fn allocated_bytes(&self) -> usize {
        self.chunks.iter().take(self.current).map(|c| c.len()).sum::<usize>() + self.offset
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new(64 * 1024) // 64KB chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_pool() {
        let mut pool: ObjectPool<i32> = ObjectPool::with_capacity(10);

        let idx1 = pool.alloc(42).unwrap();
        let idx2 = pool.alloc(100).unwrap();

        assert_eq!(*pool.get(idx1).unwrap(), 42);
        assert_eq!(*pool.get(idx2).unwrap(), 100);

        pool.free(idx1);
        assert!(pool.get(idx1).is_none()); // Stale handle
    }

    #[test]
    fn test_arena() {
        let mut arena = Arena::new(1024);

        // Allocate and check separately to avoid multiple mutable borrows
        let a_val = {
            let a = arena.alloc(42i32);
            *a
        };
        assert_eq!(a_val, 42);

        let b_val = {
            let b = arena.alloc(3.14f64);
            *b
        };
        assert!((b_val - 3.14).abs() < 1e-10);
    }
}
