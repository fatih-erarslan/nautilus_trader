//! # Spike Representation
//!
//! Ultra-compact, cache-line optimized spike structure for cortical bus communication.
//! Designed for minimum latency and maximum throughput.

use bytemuck::{Pod, Zeroable};
use std::fmt;

/// Size of a single spike in bytes (8 bytes, fits in single cache line with padding).
pub const SPIKE_SIZE: usize = 8;

/// Cache-line size for alignment (64 bytes on most modern CPUs).
pub const CACHE_LINE_SIZE: usize = 64;

/// Spikes per cache line (8 spikes fit perfectly).
pub const SPIKES_PER_CACHE_LINE: usize = CACHE_LINE_SIZE / SPIKE_SIZE;

/// Compact spike representation for cortical bus communication.
///
/// # Memory Layout
///
/// ```text
/// Offset  Size  Field         Description
/// ------  ----  -----         -----------
/// 0       4     source_id     Source neuron/pBit ID (billions of neurons)
/// 4       2     timestamp     Relative timestamp (hardware ticks, wraps)
/// 6       1     strength      Spike strength (-128 to +127, signed)
/// 7       1     routing_hint  Destination hint (minicolumn/area index)
/// ```
///
/// Total: 8 bytes (fits 8 spikes per cache line)
///
/// # Design Rationale
///
/// - **source_id (u32)**: Supports up to 4 billion neurons/pBits
/// - **timestamp (u16)**: Relative time, wraps every 65535 ticks (~1ms at 65MHz)
/// - **strength (i8)**: Signed for excitatory (+) and inhibitory (-) signals
/// - **routing_hint (u8)**: Enables fast routing to destination queues (256 partitions)
#[repr(C, align(8))]
#[derive(Copy, Clone, Default, Pod, Zeroable)]
pub struct Spike {
    /// Source neuron/pBit identifier (32-bit for billions of neurons).
    pub source_id: u32,
    /// Relative timestamp in hardware-native units (ticks/cycles).
    /// Wraps at 65535, so only valid for relative comparisons.
    pub timestamp: u16,
    /// Spike strength: positive = excitatory, negative = inhibitory.
    /// Range: -128 to +127 (normalized to [-1.0, +1.0] in processing).
    pub strength: i8,
    /// Routing hint for fast dispatch to destination queues.
    /// Used to partition spikes across multiple queues (0-255).
    pub routing_hint: u8,
}

impl Spike {
    /// Create a new spike with all fields specified.
    #[inline(always)]
    pub const fn new(source_id: u32, timestamp: u16, strength: i8, routing_hint: u8) -> Self {
        Self {
            source_id,
            timestamp,
            strength,
            routing_hint,
        }
    }

    /// Create an excitatory spike (positive strength).
    #[inline(always)]
    pub const fn excitatory(source_id: u32, timestamp: u16, routing_hint: u8) -> Self {
        Self::new(source_id, timestamp, 127, routing_hint)
    }

    /// Create an inhibitory spike (negative strength).
    #[inline(always)]
    pub const fn inhibitory(source_id: u32, timestamp: u16, routing_hint: u8) -> Self {
        Self::new(source_id, timestamp, -128, routing_hint)
    }

    /// Get normalized strength as f32 in range [-1.0, +1.0].
    #[inline(always)]
    pub fn normalized_strength(&self) -> f32 {
        self.strength as f32 / 127.0
    }

    /// Check if this is an excitatory spike.
    #[inline(always)]
    pub const fn is_excitatory(&self) -> bool {
        self.strength > 0
    }

    /// Check if this is an inhibitory spike.
    #[inline(always)]
    pub const fn is_inhibitory(&self) -> bool {
        self.strength < 0
    }

    /// Get destination queue index from routing hint.
    #[inline(always)]
    pub const fn queue_index(&self) -> usize {
        self.routing_hint as usize
    }

    /// Serialize spike to bytes (zero-copy via bytemuck).
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8; SPIKE_SIZE] {
        bytemuck::bytes_of(self).try_into().unwrap()
    }

    /// Deserialize spike from bytes (zero-copy via bytemuck).
    #[inline(always)]
    pub fn from_bytes(bytes: &[u8; SPIKE_SIZE]) -> Self {
        *bytemuck::from_bytes(bytes)
    }
}

impl fmt::Debug for Spike {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Spike")
            .field("source_id", &self.source_id)
            .field("timestamp", &self.timestamp)
            .field("strength", &self.strength)
            .field("routing_hint", &format!("0x{:02x}", self.routing_hint))
            .finish()
    }
}

impl fmt::Display for Spike {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = if self.is_excitatory() { "+" } else { "-" };
        write!(
            f,
            "Spike[{}@{}: {}{}→0x{:02x}]",
            self.source_id,
            self.timestamp,
            kind,
            self.strength.abs(),
            self.routing_hint
        )
    }
}

/// SIMD-aligned vector of spikes for batch operations.
///
/// Ensures proper alignment for AVX-512 (64-byte) operations.
#[repr(C, align(64))]
pub struct SpikeVec {
    data: Vec<Spike>,
}

impl SpikeVec {
    /// Create new spike vector with given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        // Round up to cache line boundary
        let aligned_capacity = (capacity + SPIKES_PER_CACHE_LINE - 1) / SPIKES_PER_CACHE_LINE
            * SPIKES_PER_CACHE_LINE;
        Self {
            data: Vec::with_capacity(aligned_capacity),
        }
    }

    /// Push a spike to the vector.
    #[inline(always)]
    pub fn push(&mut self, spike: Spike) {
        self.data.push(spike);
    }

    /// Get slice of spikes.
    #[inline(always)]
    pub fn as_slice(&self) -> &[Spike] {
        &self.data
    }

    /// Get mutable slice of spikes.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [Spike] {
        &mut self.data
    }

    /// Get number of spikes.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Clear all spikes.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Get raw pointer for SIMD operations.
    ///
    /// # Safety
    /// Caller must ensure proper alignment and bounds checking.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const Spike {
        self.data.as_ptr()
    }

    /// Get mutable raw pointer for SIMD operations.
    ///
    /// # Safety
    /// Caller must ensure proper alignment and bounds checking.
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut Spike {
        self.data.as_mut_ptr()
    }
}

impl Default for SpikeVec {
    fn default() -> Self {
        Self::with_capacity(1024)
    }
}

impl std::ops::Deref for SpikeVec {
    type Target = [Spike];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::DerefMut for SpikeVec {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_size() {
        assert_eq!(std::mem::size_of::<Spike>(), SPIKE_SIZE);
        assert_eq!(std::mem::align_of::<Spike>(), 8);
    }

    #[test]
    fn test_spike_creation() {
        let spike = Spike::new(12345, 100, 50, 0xAB);
        assert_eq!(spike.source_id, 12345);
        assert_eq!(spike.timestamp, 100);
        assert_eq!(spike.strength, 50);
        assert_eq!(spike.routing_hint, 0xAB);
        assert!(spike.is_excitatory());
    }

    #[test]
    fn test_excitatory_inhibitory() {
        let exc = Spike::excitatory(1, 0, 0);
        let inh = Spike::inhibitory(2, 0, 0);

        assert!(exc.is_excitatory());
        assert!(!exc.is_inhibitory());
        assert!(!inh.is_excitatory());
        assert!(inh.is_inhibitory());
    }

    #[test]
    fn test_normalized_strength() {
        let spike = Spike::new(0, 0, 127, 0);
        assert!((spike.normalized_strength() - 1.0).abs() < 0.01);

        let spike = Spike::new(0, 0, -127, 0);
        assert!((spike.normalized_strength() + 1.0).abs() < 0.01);
    }

    #[test]
    fn test_serialization() {
        let spike = Spike::new(0xDEADBEEF, 0x1234, 0x56, 0x78);
        let bytes = spike.as_bytes();
        let recovered = Spike::from_bytes(bytes);
        
        assert_eq!(spike.source_id, recovered.source_id);
        assert_eq!(spike.timestamp, recovered.timestamp);
        assert_eq!(spike.strength, recovered.strength);
        assert_eq!(spike.routing_hint, recovered.routing_hint);
    }

    #[test]
    fn test_spike_vec_alignment() {
        let vec = SpikeVec::with_capacity(100);
        let ptr = vec.as_ptr() as usize;
        // Check 8-byte alignment (Spike alignment)
        assert_eq!(ptr % 8, 0);
    }
}
