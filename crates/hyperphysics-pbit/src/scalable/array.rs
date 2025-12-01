//! Packed bit array for pBit states
//!
//! Uses 64 bits per u64 word with atomic operations for thread safety.

use std::sync::atomic::{AtomicU64, Ordering};

/// Bits per storage word
pub const BITS_PER_WORD: usize = 64;

/// Packed bit array for pBit states
///
/// Thread-safe via atomic operations. Cache-line aligned for performance.
#[repr(align(64))]
pub struct ScalablePBitArray {
    /// Packed bit storage
    words: Vec<AtomicU64>,
    /// Total number of pBits
    num_pbits: usize,
}

impl ScalablePBitArray {
    /// Create new array with all bits set to 0
    pub fn new(num_pbits: usize) -> Self {
        let num_words = (num_pbits + BITS_PER_WORD - 1) / BITS_PER_WORD;
        let words = (0..num_words).map(|_| AtomicU64::new(0)).collect();
        Self { words, num_pbits }
    }

    /// Create with random initial state
    pub fn random(num_pbits: usize, seed: u64) -> Self {
        let mut rng = fastrand::Rng::with_seed(seed);
        let num_words = (num_pbits + BITS_PER_WORD - 1) / BITS_PER_WORD;
        let words = (0..num_words)
            .map(|_| AtomicU64::new(rng.u64(..)))
            .collect();
        Self { words, num_pbits }
    }

    /// Get state of a single pBit
    #[inline(always)]
    pub fn get(&self, idx: usize) -> bool {
        debug_assert!(idx < self.num_pbits);
        let word_idx = idx / BITS_PER_WORD;
        let bit_idx = idx % BITS_PER_WORD;
        let word = self.words[word_idx].load(Ordering::Relaxed);
        (word >> bit_idx) & 1 == 1
    }

    /// Set state of a single pBit
    #[inline(always)]
    pub fn set(&self, idx: usize, value: bool) {
        debug_assert!(idx < self.num_pbits);
        let word_idx = idx / BITS_PER_WORD;
        let bit_idx = idx % BITS_PER_WORD;
        let mask = 1u64 << bit_idx;

        if value {
            self.words[word_idx].fetch_or(mask, Ordering::Relaxed);
        } else {
            self.words[word_idx].fetch_and(!mask, Ordering::Relaxed);
        }
    }

    /// Flip a single pBit atomically
    #[inline(always)]
    pub fn flip(&self, idx: usize) {
        debug_assert!(idx < self.num_pbits);
        let word_idx = idx / BITS_PER_WORD;
        let bit_idx = idx % BITS_PER_WORD;
        let mask = 1u64 << bit_idx;
        self.words[word_idx].fetch_xor(mask, Ordering::Relaxed);
    }

    /// Get spin value: false → -1.0, true → +1.0
    #[inline(always)]
    pub fn spin(&self, idx: usize) -> f32 {
        if self.get(idx) { 1.0 } else { -1.0 }
    }

    /// Get spin as i8: false → -1, true → +1 (faster for integer math)
    #[inline(always)]
    pub fn spin_i8(&self, idx: usize) -> i8 {
        if self.get(idx) { 1 } else { -1 }
    }

    /// Count number of 1 bits (population count)
    pub fn count_ones(&self) -> usize {
        self.words
            .iter()
            .map(|w| w.load(Ordering::Relaxed).count_ones() as usize)
            .sum()
    }

    /// Calculate magnetization: (N_up - N_down) / N
    pub fn magnetization(&self) -> f64 {
        let ones = self.count_ones();
        let zeros = self.num_pbits - ones;
        (ones as f64 - zeros as f64) / self.num_pbits as f64
    }

    /// Number of pBits
    #[inline]
    pub fn len(&self) -> usize {
        self.num_pbits
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_pbits == 0
    }

    /// Number of words
    #[inline]
    pub fn num_words(&self) -> usize {
        self.words.len()
    }

    /// Get raw word (for batch operations)
    #[inline]
    pub fn get_word(&self, word_idx: usize) -> u64 {
        self.words[word_idx].load(Ordering::Relaxed)
    }

    /// Set raw word
    #[inline]
    pub fn set_word(&self, word_idx: usize, value: u64) {
        self.words[word_idx].store(value, Ordering::Relaxed);
    }

    /// Clear all bits
    pub fn clear(&self) {
        for word in &self.words {
            word.store(0, Ordering::Relaxed);
        }
    }

    /// Set all bits
    pub fn fill(&self) {
        for word in &self.words {
            word.store(u64::MAX, Ordering::Relaxed);
        }
    }

    /// Create a snapshot (non-atomic copy)
    pub fn snapshot(&self) -> Vec<u64> {
        self.words
            .iter()
            .map(|w| w.load(Ordering::Relaxed))
            .collect()
    }

    /// Count state changes from snapshot
    pub fn count_changes(&self, snapshot: &[u64]) -> usize {
        self.words
            .iter()
            .zip(snapshot.iter())
            .map(|(w, s)| (w.load(Ordering::Relaxed) ^ s).count_ones() as usize)
            .sum()
    }
}

impl Clone for ScalablePBitArray {
    fn clone(&self) -> Self {
        let words = self
            .words
            .iter()
            .map(|w| AtomicU64::new(w.load(Ordering::Relaxed)))
            .collect();
        Self {
            words,
            num_pbits: self.num_pbits,
        }
    }
}

// Safety: AtomicU64 is Send + Sync
unsafe impl Send for ScalablePBitArray {}
unsafe impl Sync for ScalablePBitArray {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ops() {
        let arr = ScalablePBitArray::new(100);

        assert!(!arr.get(0));
        arr.set(0, true);
        assert!(arr.get(0));

        arr.flip(0);
        assert!(!arr.get(0));
    }

    #[test]
    fn test_across_words() {
        let arr = ScalablePBitArray::new(200);

        // Test bits in different words
        arr.set(0, true);
        arr.set(63, true);
        arr.set(64, true);
        arr.set(127, true);
        arr.set(128, true);

        assert!(arr.get(0));
        assert!(arr.get(63));
        assert!(arr.get(64));
        assert!(arr.get(127));
        assert!(arr.get(128));
        assert!(!arr.get(1));
        assert!(!arr.get(65));
    }

    #[test]
    fn test_count() {
        let arr = ScalablePBitArray::new(256);
        assert_eq!(arr.count_ones(), 0);

        for i in (0..256).step_by(2) {
            arr.set(i, true);
        }
        assert_eq!(arr.count_ones(), 128);
    }

    #[test]
    fn test_magnetization() {
        let arr = ScalablePBitArray::new(100);

        // All down: m = -1
        assert!((arr.magnetization() - (-1.0)).abs() < 0.01);

        // Set all bits individually to avoid extra bits in last word
        for i in 0..100 {
            arr.set(i, true);
        }
        // All up: m = +1
        assert!((arr.magnetization() - 1.0).abs() < 0.01, "mag = {}", arr.magnetization());
    }

    #[test]
    fn test_spin() {
        let arr = ScalablePBitArray::new(10);
        assert_eq!(arr.spin(0), -1.0);
        arr.set(0, true);
        assert_eq!(arr.spin(0), 1.0);
    }

    #[test]
    fn test_random() {
        let arr = ScalablePBitArray::random(1000, 42);
        let ones = arr.count_ones();
        // Should be roughly 50%
        assert!(ones > 400 && ones < 600, "ones = {}", ones);
    }
}
