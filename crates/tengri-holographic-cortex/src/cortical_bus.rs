//! # Ultra-Fast Cortical Bus (UFCB)
//!
//! Multi-tiered message passing infrastructure for low-latency communication
//! between pBit engines, GPU compute, and memory fabric.
//!
//! ## Tiers
//!
//! - **Tier A** (<50μs): Spike events, pinned hugepage buffers
//! - **Tier B** (<1ms): Embeddings, GPU P2P transfers
//! - **Tier C** (<10ms): Model shards, NVMe streaming

use std::sync::Arc;
use std::collections::VecDeque;
use crossbeam_queue::ArrayQueue;
use serde::{Serialize, Deserialize};

use crate::constants::*;
use crate::{CortexError, Result};

/// Bus tier for priority routing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusTier {
    /// Ultra-low latency spikes (<50μs)
    TierA,
    /// Medium latency embeddings (<1ms)
    TierB,
    /// Higher latency shards (<10ms)
    TierC,
}

/// Spike packet for Tier A
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikePacket {
    /// Source engine ID
    pub source_engine: usize,
    /// Spike timestamps (tick)
    pub timestamp: u64,
    /// Active node indices
    pub node_ids: Vec<u64>,
    /// Optional metadata
    pub metadata: Option<Vec<u8>>,
}

impl SpikePacket {
    /// Create new spike packet
    pub fn new(source: usize, timestamp: u64, node_ids: Vec<u64>) -> Self {
        Self {
            source_engine: source,
            timestamp,
            node_ids,
            metadata: None,
        }
    }
    
    /// Get packet size in bytes
    pub fn size_bytes(&self) -> usize {
        8 + 8 + self.node_ids.len() * 8 + self.metadata.as_ref().map_or(0, |m| m.len())
    }
}

/// Embedding packet for Tier B
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingPacket {
    /// Source engine ID
    pub source_engine: usize,
    /// Timestamp
    pub timestamp: u64,
    /// Embedding dimension
    pub dimension: usize,
    /// Embedding data (flattened)
    pub data: Vec<f32>,
}

/// Shard request for Tier C
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardRequest {
    /// Shard ID
    pub shard_id: u64,
    /// Offset within shard
    pub offset: usize,
    /// Length in bytes
    pub length: usize,
    /// Priority (lower = higher priority)
    pub priority: u8,
}

/// Ultra-Fast Cortical Bus
pub struct CorticalBus {
    /// Tier A queue (spikes)
    tier_a: Arc<ArrayQueue<SpikePacket>>,
    /// Tier B queue (embeddings)
    tier_b: Arc<ArrayQueue<EmbeddingPacket>>,
    /// Tier C queue (shard requests)
    tier_c: Arc<ArrayQueue<ShardRequest>>,
    /// Statistics
    stats: BusStats,
}

/// Bus statistics
#[derive(Debug, Clone, Default)]
pub struct BusStats {
    /// Tier A packets sent
    pub tier_a_sent: u64,
    /// Tier A packets received
    pub tier_a_recv: u64,
    /// Tier B packets sent
    pub tier_b_sent: u64,
    /// Tier B packets received
    pub tier_b_recv: u64,
    /// Tier C requests sent
    pub tier_c_sent: u64,
    /// Tier C requests completed
    pub tier_c_completed: u64,
    /// Total bytes transferred
    pub total_bytes: u64,
}

impl CorticalBus {
    /// Create new cortical bus with default capacity
    pub fn new() -> Self {
        Self::with_capacity(
            BUS_TIER_A_SIZE / SPIKE_PACKET_SIZE,
            256 * 1024, // 256K embedding packets
            64 * 1024,  // 64K shard requests
        )
    }
    
    /// Create with custom capacities
    pub fn with_capacity(tier_a_cap: usize, tier_b_cap: usize, tier_c_cap: usize) -> Self {
        Self {
            tier_a: Arc::new(ArrayQueue::new(tier_a_cap)),
            tier_b: Arc::new(ArrayQueue::new(tier_b_cap)),
            tier_c: Arc::new(ArrayQueue::new(tier_c_cap)),
            stats: BusStats::default(),
        }
    }
    
    /// Publish spike packet to Tier A
    pub fn publish_spikes(&mut self, packet: SpikePacket) -> Result<()> {
        let bytes = packet.size_bytes();
        
        match self.tier_a.push(packet) {
            Ok(()) => {
                self.stats.tier_a_sent += 1;
                self.stats.total_bytes += bytes as u64;
                Ok(())
            }
            Err(_) => Err(CortexError::BusError("Tier A queue full".into())),
        }
    }
    
    /// Receive spike packet from Tier A
    pub fn receive_spikes(&mut self) -> Option<SpikePacket> {
        self.tier_a.pop().map(|p| {
            self.stats.tier_a_recv += 1;
            p
        })
    }
    
    /// Publish embedding to Tier B
    pub fn publish_embedding(&mut self, packet: EmbeddingPacket) -> Result<()> {
        let bytes = packet.data.len() * 4 + 24;
        
        match self.tier_b.push(packet) {
            Ok(()) => {
                self.stats.tier_b_sent += 1;
                self.stats.total_bytes += bytes as u64;
                Ok(())
            }
            Err(_) => Err(CortexError::BusError("Tier B queue full".into())),
        }
    }
    
    /// Receive embedding from Tier B
    pub fn receive_embedding(&mut self) -> Option<EmbeddingPacket> {
        self.tier_b.pop().map(|p| {
            self.stats.tier_b_recv += 1;
            p
        })
    }
    
    /// Submit shard request to Tier C
    pub fn request_shard(&mut self, request: ShardRequest) -> Result<()> {
        match self.tier_c.push(request) {
            Ok(()) => {
                self.stats.tier_c_sent += 1;
                Ok(())
            }
            Err(_) => Err(CortexError::BusError("Tier C queue full".into())),
        }
    }
    
    /// Get next shard request from Tier C
    pub fn next_shard_request(&mut self) -> Option<ShardRequest> {
        self.tier_c.pop()
    }
    
    /// Mark shard request as completed
    pub fn complete_shard(&mut self, bytes: usize) {
        self.stats.tier_c_completed += 1;
        self.stats.total_bytes += bytes as u64;
    }
    
    /// Get queue lengths
    pub fn queue_lengths(&self) -> (usize, usize, usize) {
        (self.tier_a.len(), self.tier_b.len(), self.tier_c.len())
    }
    
    /// Check if any tier is backpressured (>80% full)
    pub fn is_backpressured(&self) -> bool {
        let (a, b, c) = self.queue_lengths();
        let (cap_a, cap_b, cap_c) = (
            self.tier_a.capacity(),
            self.tier_b.capacity(),
            self.tier_c.capacity(),
        );
        
        a > cap_a * 8 / 10 || b > cap_b * 8 / 10 || c > cap_c * 8 / 10
    }
    
    /// Get statistics
    pub fn stats(&self) -> &BusStats {
        &self.stats
    }
    
    /// Drain all queues (for shutdown)
    pub fn drain(&mut self) {
        while self.tier_a.pop().is_some() {}
        while self.tier_b.pop().is_some() {}
        while self.tier_c.pop().is_some() {}
    }
    
    /// Get clone of Tier A queue for multi-threaded access
    pub fn tier_a_handle(&self) -> Arc<ArrayQueue<SpikePacket>> {
        Arc::clone(&self.tier_a)
    }
    
    /// Get clone of Tier B queue
    pub fn tier_b_handle(&self) -> Arc<ArrayQueue<EmbeddingPacket>> {
        Arc::clone(&self.tier_b)
    }
}

impl Default for CorticalBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bus_creation() {
        let bus = CorticalBus::new();
        let (a, b, c) = bus.queue_lengths();
        assert_eq!(a, 0);
        assert_eq!(b, 0);
        assert_eq!(c, 0);
    }
    
    #[test]
    fn test_spike_roundtrip() {
        let mut bus = CorticalBus::new();
        
        let packet = SpikePacket::new(0, 100, vec![1, 2, 3, 4, 5]);
        bus.publish_spikes(packet).unwrap();
        
        let received = bus.receive_spikes().unwrap();
        assert_eq!(received.source_engine, 0);
        assert_eq!(received.timestamp, 100);
        assert_eq!(received.node_ids.len(), 5);
    }
    
    #[test]
    fn test_embedding_roundtrip() {
        let mut bus = CorticalBus::new();
        
        let packet = EmbeddingPacket {
            source_engine: 1,
            timestamp: 200,
            dimension: 11,
            data: vec![0.1; 11],
        };
        bus.publish_embedding(packet).unwrap();
        
        let received = bus.receive_embedding().unwrap();
        assert_eq!(received.source_engine, 1);
        assert_eq!(received.dimension, 11);
    }
    
    #[test]
    fn test_stats_tracking() {
        let mut bus = CorticalBus::new();
        
        for i in 0..10 {
            let packet = SpikePacket::new(0, i, vec![i]);
            bus.publish_spikes(packet).unwrap();
        }
        
        assert_eq!(bus.stats().tier_a_sent, 10);
        
        for _ in 0..5 {
            bus.receive_spikes();
        }
        
        assert_eq!(bus.stats().tier_a_recv, 5);
    }
}
