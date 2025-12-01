//! # Packed Bit Array
//!
//! Ultra-compact storage for pBit states using 64 bits per word.
//! Optimized for cache efficiency and SIMD operations.

use std::sync::atomic::{AtomicU64, Ordering};

/// Bits per storage word
pub const BITS_PER_WORD: usize = 64;

/// Cache line size for alignment
const CACHE_LINE_SIZE: usize = 64;

/// Packed bit array for pBit states
///
/// Uses atomic operations for thread-safe access.
/// Memory layout is cache-line aligned for optimal performance.
#[repr(align(64))]
pub struct PackedPBitArray {
    /// Packed bit storage (64 pBits per word)
    words: Vec<AtomicU64>,
    /// Total number of pBits
    num_bits: usize,
}

impl PackedPBitArray {
    /// Create a new packed array with all bits set to 0
    pub fn new(num_bits: usize) -> Self {
        let num_words = (num_bits + BITS_PER_WORD - 1) / BITS_PER_WORD;
        let words = (0..num_words)
            .map(|_| AtomicU64::new(0))
            .collect();
        
        Self { words, num_bits }
    }

    /// Create with random initial state
    pub fn random(num_bits: usize, seed: u64) -> Self {
        let mut rng = fastrand::Rng::with_seed(seed);
        let num_words = (num_bits + BITS_PER_WORD - 1) / BITS_PER_WORD;
        let words = (0..num_words)
            .map(|_| AtomicU64::new(rng.u64(..)))
            .collect();
        
        Self { words, num_bits }
    }

    /// Get the state of a single bit
    #[inline(always)]
    pub fn get(&self, idx: usize) -> bool {
        debug_assert!(idx < self.num_bits);
        let word_idx = idx / BITS_PER_WORD;
        let bit_idx = idx % BITS_PER_WORD;
        let word = self.words[word_idx].load(Ordering::Relaxed);
        (word >> bit_idx) & 1 == 1
    }

    /// Set the state of a single bit
    #[inline(always)]
    pub fn set(&self, idx: usize, value: bool) {
        debug_assert!(idx < self.num_bits);
        let word_idx = idx / BITS_PER_WORD;
        let bit_idx = idx % BITS_PER_WORD;
        let mask = 1u64 << bit_idx;
        
        if value {
            self.words[word_idx].fetch_or(mask, Ordering::Relaxed);
        } else {
            self.words[word_idx].fetch_and(!mask, Ordering::Relaxed);
        }
    }

    /// Flip a single bit atomically
    #[inline(always)]
    pub fn flip(&self, idx: usize) {
        debug_assert!(idx < self.num_bits);
        let word_idx = idx / BITS_PER_WORD;
        let bit_idx = idx % BITS_PER_WORD;
        let mask = 1u64 << bit_idx;
        self.words[word_idx].fetch_xor(mask, Ordering::Relaxed);
    }

    /// Get spin value (-1 or +1)
    #[inline(always)]
    pub fn spin(&self, idx: usize) -> f32 {
        if self.get(idx) { 1.0 } else { -1.0 }
    }

    /// Count total number of 1 bits (population count)
    pub fn count_ones(&self) -> usize {
        self.words
            .iter()
            .map(|w| w.load(Ordering::Relaxed).count_ones() as usize)
            .sum()
    }

    /// Count number of 0 bits
    pub fn count_zeros(&self) -> usize {
        self.num_bits - self.count_ones()
    }

    /// Set all bits to 0
    pub fn clear(&self) {
        for word in &self.words {
            word.store(0, Ordering::Relaxed);
        }
    }

    /// Set all bits to 1
    pub fn fill(&self) {
        for word in &self.words {
            word.store(u64::MAX, Ordering::Relaxed);
        }
    }

    /// Get raw word for batch operations
    #[inline(always)]
    pub fn get_word(&self, word_idx: usize) -> u64 {
        self.words[word_idx].load(Ordering::Relaxed)
    }

    /// Set raw word for batch operations
    #[inline(always)]
    pub fn set_word(&self, word_idx: usize, value: u64) {
        self.words[word_idx].store(value, Ordering::Relaxed);
    }

    /// Number of bits
    #[inline]
    pub fn len(&self) -> usize {
        self.num_bits
    }

    /// Number of words
    #[inline]
    pub fn num_words(&self) -> usize {
        self.words.len()
    }

    /// Iterate over word indices for a range of bits
    #[inline]
    pub fn word_range(&self, start_bit: usize, end_bit: usize) -> std::ops::Range<usize> {
        let start_word = start_bit / BITS_PER_WORD;
        let end_word = (end_bit + BITS_PER_WORD - 1) / BITS_PER_WORD;
        start_word..end_word.min(self.words.len())
    }

    /// XOR with another array (useful for detecting changes)
    pub fn xor_count(&self, other: &PackedPBitArray) -> usize {
        assert_eq!(self.num_bits, other.num_bits);
        self.words
            .iter()
            .zip(other.words.iter())
            .map(|(a, b)| {
                let diff = a.load(Ordering::Relaxed) ^ b.load(Ordering::Relaxed);
                diff.count_ones() as usize
            })
            .sum()
    }

    /// Clone state to a new array
    pub fn snapshot(&self) -> PackedPBitArray {
        let words = self.words
            .iter()
            .map(|w| AtomicU64::new(w.load(Ordering::Relaxed)))
            .collect();
        
        PackedPBitArray {
            words,
            num_bits: self.num_bits,
        }
    }
}

impl Clone for PackedPBitArray {
    fn clone(&self) -> Self {
        self.snapshot()
    }
}

unsafe impl Send for PackedPBitArray {}
unsafe impl Sync for PackedPBitArray {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let arr = PackedPBitArray::new(100);
        
        assert!(!arr.get(0));
        arr.set(0, true);
        assert!(arr.get(0));
        
        arr.flip(0);
        assert!(!arr.get(0));
    }

    #[test]
    fn test_count() {
        let arr = PackedPBitArray::new(128);
        assert_eq!(arr.count_ones(), 0);
        
        arr.set(0, true);
        arr.set(63, true);
        arr.set(64, true);
        arr.set(127, true);
        
        assert_eq!(arr.count_ones(), 4);
    }

    #[test]
    fn test_large_array() {
        let arr = PackedPBitArray::new(1_000_000);
        
        // Set every 1000th bit
        for i in (0..1_000_000).step_by(1000) {
            arr.set(i, true);
        }
        
        assert_eq!(arr.count_ones(), 1000);
    }

    #[test]
    fn test_spin() {
        let arr = PackedPBitArray::new(10);
        
        assert_eq!(arr.spin(0), -1.0);
        arr.set(0, true);
        assert_eq!(arr.spin(0), 1.0);
    }

    #[test]
    fn test_random_init() {
        let arr = PackedPBitArray::random(1000, 42);
        
        // Should have roughly 50% ones
        let ones = arr.count_ones();
        assert!(ones > 400 && ones < 600, "Expected ~500 ones, got {}", ones);
    }
}
