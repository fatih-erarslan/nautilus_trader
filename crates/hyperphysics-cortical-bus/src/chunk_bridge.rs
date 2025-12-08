//! # ChunkProcessor Bridge for Cortical Bus
//!
//! Integrates `hyperphysics-geometry::ChunkProcessor` for temporal spike hierarchy.
//!
//! ## Temporal Scales
//!
//! The ChunkProcessor implements a multi-scale temporal hierarchy:
//!
//! | Timescale | Description | Use in Cortical Bus |
//! |-----------|-------------|---------------------|
//! | 1ms | Ultra-fast spikes | Individual spike events |
//! | 10ms | Fast patterns | Burst detection |
//! | 100ms | Standard processing | Pattern recognition |
//! | 1s | Slow dynamics | State tracking |
//! | 10s | Very slow | Context memory |
//!
//! ## Integration with Spike Router
//!
//! The ChunkProcessor bridges raw spike events to hierarchical representations:
//!
//! ```text
//! Raw Spikes (1ms) → ChunkProcessor → Hierarchical Chunks
//!       ↓                                    ↓
//! Ring Buffer        →              Pattern Memory (LSH/HNSW)
//! ```
//!
//! ## SOC-Aware Chunking
//!
//! When SOC statistics are available, chunk boundaries adapt to criticality:
//! - At criticality (σ ≈ 1.0): Finer temporal resolution
//! - Sub-critical (σ < 1.0): Coarser chunking
//! - Super-critical (σ > 1.0): Adaptive chunking based on avalanche sizes

use crate::spike::{Spike, SpikeVec};
use crate::error::{CorticalError, Result};

#[cfg(feature = "geometry")]
use hyperphysics_geometry::{
    ChunkProcessor, ChunkConfig, TimescaleLevel, ChunkRepresentation,
    SOCStats, ProcessorStats,
};

/// Configuration for the ChunkProcessor bridge
#[derive(Debug, Clone)]
pub struct ChunkBridgeConfig {
    /// Maximum spikes to buffer before forcing chunk processing
    pub max_buffer_size: usize,
    /// Time window for chunk accumulation (ms)
    pub chunk_window_ms: f64,
    /// Enable SOC-aware adaptive chunking
    pub soc_adaptive: bool,
    /// Minimum chunk size for SOC adaptation
    pub min_chunk_size: usize,
    /// Maximum chunk size for SOC adaptation
    pub max_chunk_size: usize,
}

impl Default for ChunkBridgeConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 1024,
            chunk_window_ms: 10.0, // 10ms default chunk window
            soc_adaptive: true,
            min_chunk_size: 8,
            max_chunk_size: 512,
        }
    }
}

/// Bridge between raw spikes and ChunkProcessor
///
/// Accumulates spikes from the cortical bus and processes them
/// into hierarchical chunk representations.
#[cfg(feature = "geometry")]
pub struct ChunkProcessorBridge {
    /// Configuration
    config: ChunkBridgeConfig,
    /// Underlying ChunkProcessor from geometry crate
    processor: ChunkProcessor,
    /// Spike buffer for accumulation
    spike_buffer: Vec<SpikeData>,
    /// Current buffer start time
    buffer_start_time: f64,
    /// Latest SOC statistics (if available)
    soc_stats: Option<SOCStats>,
    /// Processing statistics
    stats: BridgeStats,
}

/// Internal spike data structure for chunk processing
#[cfg(feature = "geometry")]
#[derive(Debug, Clone)]
struct SpikeData {
    source_id: u32,
    timestamp: f64,
    strength: f32,
    routing_hint: u8,
}

/// Bridge processing statistics
#[cfg(feature = "geometry")]
#[derive(Debug, Clone, Default)]
pub struct BridgeStats {
    /// Total spikes processed
    pub spikes_processed: u64,
    /// Total chunks generated
    pub chunks_generated: u64,
    /// Average spikes per chunk
    pub avg_spikes_per_chunk: f64,
    /// Current buffer utilization
    pub buffer_utilization: f64,
    /// SOC-adaptive chunk resizes
    pub soc_adaptations: u64,
}

#[cfg(feature = "geometry")]
impl ChunkProcessorBridge {
    /// Create new bridge with default configuration
    pub fn new() -> Self {
        Self::with_config(ChunkBridgeConfig::default())
    }

    /// Create bridge with custom configuration
    pub fn with_config(config: ChunkBridgeConfig) -> Self {
        let chunk_config = ChunkConfig::default();
        let processor = ChunkProcessor::new(chunk_config);

        Self {
            config,
            processor,
            spike_buffer: Vec::with_capacity(1024),
            buffer_start_time: 0.0,
            soc_stats: None,
            stats: BridgeStats::default(),
        }
    }

    /// Update SOC statistics for adaptive chunking
    pub fn update_soc_stats(&mut self, stats: SOCStats) {
        self.soc_stats = Some(stats);
    }

    /// Add a spike to the buffer
    ///
    /// Returns a chunk if the buffer triggers processing.
    pub fn add_spike(&mut self, spike: &Spike, current_time: f64) -> Option<ChunkRepresentation> {
        // Initialize buffer start time if needed
        if self.spike_buffer.is_empty() {
            self.buffer_start_time = current_time;
        }

        // Add spike to buffer
        self.spike_buffer.push(SpikeData {
            source_id: spike.source_id,
            timestamp: current_time,
            strength: spike.normalized_strength(),
            routing_hint: spike.routing_hint,
        });

        self.stats.spikes_processed += 1;

        // Determine chunk window based on SOC state
        let effective_window = self.compute_effective_window();

        // Check if we should process
        let time_elapsed = current_time - self.buffer_start_time;
        let buffer_full = self.spike_buffer.len() >= self.config.max_buffer_size;
        let window_complete = time_elapsed >= effective_window;

        if buffer_full || window_complete {
            return self.process_buffer();
        }

        None
    }

    /// Add multiple spikes from a SpikeVec
    pub fn add_spikes(&mut self, spikes: &SpikeVec, current_time: f64) -> Vec<ChunkRepresentation> {
        let mut chunks = Vec::new();

        for spike in spikes.as_slice() {
            // Estimate timestamp from relative offset
            let spike_time = current_time + (spike.timestamp as f64 / 65536.0) * self.config.chunk_window_ms;

            if let Some(chunk) = self.add_spike(spike, spike_time) {
                chunks.push(chunk);
            }
        }

        chunks
    }

    /// Compute effective chunk window based on SOC state
    fn compute_effective_window(&mut self) -> f64 {
        if !self.config.soc_adaptive {
            return self.config.chunk_window_ms;
        }

        let base_window = self.config.chunk_window_ms;

        if let Some(stats) = &self.soc_stats {
            // At criticality, use finer temporal resolution
            let sigma_factor = if stats.is_critical {
                0.5 // Shorter windows at criticality
            } else if stats.sigma_measured < 1.0 {
                // Sub-critical: larger windows
                1.0 + (1.0 - stats.sigma_measured).min(0.5)
            } else {
                // Super-critical: adapt based on avalanche size
                let avalanche_factor = (stats.avg_avalanche_size / 10.0).clamp(0.5, 2.0);
                avalanche_factor
            };

            self.stats.soc_adaptations += 1;
            base_window * sigma_factor
        } else {
            base_window
        }
    }

    /// Process the current spike buffer into a chunk
    fn process_buffer(&mut self) -> Option<ChunkRepresentation> {
        if self.spike_buffer.is_empty() {
            return None;
        }

        // Convert spike data to format suitable for ChunkProcessor
        let spike_times: Vec<f64> = self.spike_buffer.iter().map(|s| s.timestamp).collect();
        let spike_strengths: Vec<f32> = self.spike_buffer.iter().map(|s| s.strength).collect();

        // Process through geometry crate's ChunkProcessor
        let chunk = self.processor.process_spike_train(&spike_times, &spike_strengths);

        // Update statistics
        self.stats.chunks_generated += 1;
        let total_spikes = self.stats.spikes_processed as f64;
        let total_chunks = self.stats.chunks_generated as f64;
        self.stats.avg_spikes_per_chunk = total_spikes / total_chunks;
        self.stats.buffer_utilization =
            self.spike_buffer.len() as f64 / self.config.max_buffer_size as f64;

        // Clear buffer
        self.spike_buffer.clear();

        chunk
    }

    /// Force processing of any remaining spikes
    pub fn flush(&mut self) -> Option<ChunkRepresentation> {
        self.process_buffer()
    }

    /// Get processing statistics
    pub fn stats(&self) -> &BridgeStats {
        &self.stats
    }

    /// Get processor statistics from underlying ChunkProcessor
    pub fn processor_stats(&self) -> ProcessorStats {
        self.processor.stats()
    }

    /// Get current buffer size
    pub fn buffer_size(&self) -> usize {
        self.spike_buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.spike_buffer.is_empty()
    }
}

#[cfg(feature = "geometry")]
impl Default for ChunkProcessorBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Temporal scale mapper for cortical bus timescales
///
/// Maps cortical bus timing (hardware ticks) to ChunkProcessor timescales.
#[derive(Debug, Clone)]
pub struct TimescaleMapper {
    /// Hardware ticks per millisecond
    pub ticks_per_ms: f64,
    /// Current timescale level
    pub current_level: u8,
}

impl Default for TimescaleMapper {
    fn default() -> Self {
        Self {
            ticks_per_ms: 65.536, // 65.536 kHz tick rate (16-bit timestamp wraps at ~1ms)
            current_level: 1, // 10ms default
        }
    }
}

impl TimescaleMapper {
    /// Create mapper with custom tick rate
    pub fn with_tick_rate(ticks_per_ms: f64) -> Self {
        Self {
            ticks_per_ms,
            current_level: 1,
        }
    }

    /// Convert hardware ticks to milliseconds
    pub fn ticks_to_ms(&self, ticks: u16) -> f64 {
        ticks as f64 / self.ticks_per_ms
    }

    /// Convert milliseconds to hardware ticks
    pub fn ms_to_ticks(&self, ms: f64) -> u16 {
        (ms * self.ticks_per_ms).round() as u16
    }

    /// Get current timescale in milliseconds
    #[cfg(feature = "geometry")]
    pub fn current_scale_ms(&self) -> f64 {
        TimescaleLevel::from_level(self.current_level).window_ms()
    }

    /// Advance to next coarser timescale
    pub fn advance_level(&mut self) {
        if self.current_level < 4 {
            self.current_level += 1;
        }
    }

    /// Return to finer timescale
    pub fn retreat_level(&mut self) {
        if self.current_level > 0 {
            self.current_level -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timescale_mapper() {
        let mapper = TimescaleMapper::default();

        // Test tick conversion
        let ticks: u16 = 6554; // ~100ms at default rate
        let ms = mapper.ticks_to_ms(ticks);
        assert!((ms - 100.0).abs() < 1.0);

        // Round trip
        let back_to_ticks = mapper.ms_to_ticks(ms);
        assert_eq!(ticks, back_to_ticks);
    }

    #[test]
    fn test_chunk_bridge_config() {
        let config = ChunkBridgeConfig::default();
        assert_eq!(config.chunk_window_ms, 10.0);
        assert!(config.soc_adaptive);
        assert_eq!(config.max_buffer_size, 1024);
    }

    #[cfg(feature = "geometry")]
    #[test]
    fn test_chunk_bridge_creation() {
        let bridge = ChunkProcessorBridge::new();
        assert!(bridge.is_empty());
        assert_eq!(bridge.buffer_size(), 0);
    }

    #[cfg(feature = "geometry")]
    #[test]
    fn test_spike_accumulation() {
        let mut bridge = ChunkProcessorBridge::new();

        // Add some spikes
        for i in 0..10 {
            let spike = Spike::new(i as u32, i as u16, 100, 0);
            bridge.add_spike(&spike, i as f64 * 0.1);
        }

        assert_eq!(bridge.buffer_size(), 10);
        assert_eq!(bridge.stats().spikes_processed, 10);
    }

    #[cfg(feature = "geometry")]
    #[test]
    fn test_soc_adaptive_window() {
        let mut bridge = ChunkProcessorBridge::new();

        // At criticality, window should shrink
        let critical_stats = SOCStats {
            sigma_measured: 1.0,
            sigma_target: 1.0,
            is_critical: true,
            ..Default::default()
        };
        bridge.update_soc_stats(critical_stats);

        let window = bridge.compute_effective_window();
        assert!(window < bridge.config.chunk_window_ms);

        // Sub-critical, window should grow
        let subcritical_stats = SOCStats {
            sigma_measured: 0.7,
            sigma_target: 1.0,
            is_critical: false,
            ..Default::default()
        };
        bridge.update_soc_stats(subcritical_stats);

        let window = bridge.compute_effective_window();
        assert!(window > bridge.config.chunk_window_ms);
    }
}
