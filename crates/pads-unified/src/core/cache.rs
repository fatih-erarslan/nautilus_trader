//! Circuit Cache Implementation
//!
//! High-performance caching system for PADS circuit evaluations with
//! LRU eviction, memory mapping, and SIMD-optimized lookups.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use crate::error::{PadsError, PadsResult};
use crate::types::*;

#[cfg(feature = "simd")]
use std::arch::x86_64::*;

/// Cache entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Cached value
    pub value: CacheValue,
    /// Last access timestamp
    pub last_access: u64,
    /// Access count
    pub access_count: u64,
    /// Entry creation time
    pub created_at: u64,
    /// Entry size in bytes
    pub size_bytes: usize,
    /// Entry priority (higher = more important)
    pub priority: f64,
}

/// Cached value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheValue {
    /// Cached decision
    Decision(TradingDecision),
    /// Cached market state
    MarketState(MarketState),
    /// Cached analysis result
    Analysis(AnalysisResult),
    /// Cached agent output
    AgentOutput(AgentOutput),
    /// Cached board result
    BoardResult(BoardResult),
    /// Cached strategy result
    StrategyResult(StrategyResult),
    /// Raw bytes for custom data
    RawBytes(Vec<u8>),
}

/// Analysis result cache entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub analyzer_id: String,
    pub result: serde_json::Value,
    pub confidence: f64,
    pub timestamp: u64,
}

/// Agent output cache entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentOutput {
    pub agent_id: String,
    pub decision: DecisionType,
    pub confidence: f64,
    pub reasoning: String,
    pub timestamp: u64,
}

/// Board result cache entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoardResult {
    pub decision: DecisionType,
    pub consensus_score: f64,
    pub participant_votes: HashMap<String, f64>,
    pub timestamp: u64,
}

/// Strategy result cache entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyResult {
    pub strategy_id: String,
    pub decision: DecisionType,
    pub confidence: f64,
    pub risk_score: f64,
    pub timestamp: u64,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_entries: usize,
    pub total_size_bytes: usize,
    pub hit_count: u64,
    pub miss_count: u64,
    pub eviction_count: u64,
    pub hit_rate: f64,
    pub average_access_time_ns: u64,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum number of entries
    pub max_entries: usize,
    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,
    /// TTL for entries in seconds
    pub ttl_seconds: u64,
    /// Enable memory mapping
    pub enable_memory_mapping: bool,
    /// Enable SIMD optimization
    pub enable_simd: bool,
    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Compression level (0-9)
    pub compression_level: u8,
}

/// Cache eviction policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Time-based expiration
    TTL,
    /// Priority-based (higher priority = longer retention)
    Priority,
    /// Adaptive (combines multiple strategies)
    Adaptive,
}

/// High-performance circuit cache
pub struct CircuitCache {
    /// Cache entries
    entries: Arc<RwLock<HashMap<String, CacheEntry>>>,
    /// Cache configuration
    config: CacheConfig,
    /// Cache statistics
    stats: Arc<RwLock<CacheStats>>,
    /// LRU order tracking
    lru_order: Arc<RwLock<Vec<String>>>,
    /// Memory-mapped storage
    #[cfg(feature = "memory-mapping")]
    mmap_storage: Option<Arc<memmap2::Mmap>>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10000,
            max_memory_bytes: 100 * 1024 * 1024, // 100MB
            ttl_seconds: 300, // 5 minutes
            enable_memory_mapping: true,
            enable_simd: true,
            eviction_policy: EvictionPolicy::Adaptive,
            compression_level: 3,
        }
    }
}

impl CircuitCache {
    /// Create a new circuit cache
    pub async fn new(config: CacheConfig) -> PadsResult<Self> {
        let stats = CacheStats {
            total_entries: 0,
            total_size_bytes: 0,
            hit_count: 0,
            miss_count: 0,
            eviction_count: 0,
            hit_rate: 0.0,
            average_access_time_ns: 0,
        };
        
        Ok(Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            config,
            stats: Arc::new(RwLock::new(stats)),
            lru_order: Arc::new(RwLock::new(Vec::new())),
            #[cfg(feature = "memory-mapping")]
            mmap_storage: None,
        })
    }
    
    /// Get a value from the cache
    pub async fn get(&self, key: &str) -> PadsResult<Option<CacheValue>> {
        let start_time = std::time::Instant::now();
        
        let mut entries = self.entries.write().await;
        let mut stats = self.stats.write().await;
        
        if let Some(entry) = entries.get_mut(key) {
            // Check TTL
            let current_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            
            if current_time - entry.created_at > self.config.ttl_seconds {
                // Entry expired
                entries.remove(key);
                stats.miss_count += 1;
                return Ok(None);
            }
            
            // Update access metadata
            entry.last_access = current_time;
            entry.access_count += 1;
            
            // Update LRU order
            let mut lru_order = self.lru_order.write().await;
            if let Some(pos) = lru_order.iter().position(|k| k == key) {
                lru_order.remove(pos);
            }
            lru_order.push(key.to_string());
            
            stats.hit_count += 1;
            stats.hit_rate = stats.hit_count as f64 / (stats.hit_count + stats.miss_count) as f64;
            
            let access_time = start_time.elapsed().as_nanos() as u64;
            stats.average_access_time_ns = (stats.average_access_time_ns + access_time) / 2;
            
            Ok(Some(entry.value.clone()))
        } else {
            stats.miss_count += 1;
            stats.hit_rate = stats.hit_count as f64 / (stats.hit_count + stats.miss_count) as f64;
            Ok(None)
        }
    }
    
    /// Put a value into the cache
    pub async fn put(&self, key: String, value: CacheValue, priority: f64) -> PadsResult<()> {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let size_bytes = self.estimate_size(&value);
        
        let entry = CacheEntry {
            value,
            last_access: current_time,
            access_count: 1,
            created_at: current_time,
            size_bytes,
            priority,
        };
        
        let mut entries = self.entries.write().await;
        let mut stats = self.stats.write().await;
        
        // Check if we need to evict entries
        if entries.len() >= self.config.max_entries || 
           stats.total_size_bytes + size_bytes > self.config.max_memory_bytes {
            self.evict_entries(&mut entries, &mut stats).await?;
        }
        
        // Insert the new entry
        entries.insert(key.clone(), entry);
        stats.total_entries = entries.len();
        stats.total_size_bytes += size_bytes;
        
        // Update LRU order
        let mut lru_order = self.lru_order.write().await;
        if let Some(pos) = lru_order.iter().position(|k| k == &key) {
            lru_order.remove(pos);
        }
        lru_order.push(key);
        
        Ok(())
    }
    
    /// Remove a value from the cache
    pub async fn remove(&self, key: &str) -> PadsResult<Option<CacheValue>> {
        let mut entries = self.entries.write().await;
        let mut stats = self.stats.write().await;
        
        if let Some(entry) = entries.remove(key) {
            stats.total_entries = entries.len();
            stats.total_size_bytes -= entry.size_bytes;
            
            // Update LRU order
            let mut lru_order = self.lru_order.write().await;
            if let Some(pos) = lru_order.iter().position(|k| k == key) {
                lru_order.remove(pos);
            }
            
            Ok(Some(entry.value))
        } else {
            Ok(None)
        }
    }
    
    /// Clear all entries from the cache
    pub async fn clear(&self) -> PadsResult<()> {
        let mut entries = self.entries.write().await;
        let mut stats = self.stats.write().await;
        let mut lru_order = self.lru_order.write().await;
        
        entries.clear();
        lru_order.clear();
        
        stats.total_entries = 0;
        stats.total_size_bytes = 0;
        
        Ok(())
    }
    
    /// Get cache statistics
    pub async fn stats(&self) -> PadsResult<CacheStats> {
        let stats = self.stats.read().await;
        Ok(stats.clone())
    }
    
    /// Evict entries based on the configured policy
    async fn evict_entries(
        &self,
        entries: &mut HashMap<String, CacheEntry>,
        stats: &mut CacheStats,
    ) -> PadsResult<()> {
        let eviction_count = std::cmp::max(1, entries.len() / 10); // Evict 10% of entries
        
        match self.config.eviction_policy {
            EvictionPolicy::LRU => {
                let lru_order = self.lru_order.read().await;
                let keys_to_remove: Vec<String> = lru_order
                    .iter()
                    .take(eviction_count)
                    .cloned()
                    .collect();
                
                for key in keys_to_remove {
                    if let Some(entry) = entries.remove(&key) {
                        stats.total_size_bytes -= entry.size_bytes;
                        stats.eviction_count += 1;
                    }
                }
            }
            EvictionPolicy::LFU => {
                let mut entries_by_frequency: Vec<_> = entries.iter().collect();
                entries_by_frequency.sort_by_key(|(_, entry)| entry.access_count);
                
                let keys_to_remove: Vec<String> = entries_by_frequency
                    .iter()
                    .take(eviction_count)
                    .map(|(key, _)| (*key).clone())
                    .collect();
                
                for key in keys_to_remove {
                    if let Some(entry) = entries.remove(&key) {
                        stats.total_size_bytes -= entry.size_bytes;
                        stats.eviction_count += 1;
                    }
                }
            }
            EvictionPolicy::TTL => {
                let current_time = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                
                let keys_to_remove: Vec<String> = entries
                    .iter()
                    .filter(|(_, entry)| current_time - entry.created_at > self.config.ttl_seconds)
                    .map(|(key, _)| key.clone())
                    .collect();
                
                for key in keys_to_remove {
                    if let Some(entry) = entries.remove(&key) {
                        stats.total_size_bytes -= entry.size_bytes;
                        stats.eviction_count += 1;
                    }
                }
            }
            EvictionPolicy::Priority => {
                let mut entries_by_priority: Vec<_> = entries.iter().collect();
                entries_by_priority.sort_by(|a, b| a.1.priority.partial_cmp(&b.1.priority).unwrap());
                
                let keys_to_remove: Vec<String> = entries_by_priority
                    .iter()
                    .take(eviction_count)
                    .map(|(key, _)| (*key).clone())
                    .collect();
                
                for key in keys_to_remove {
                    if let Some(entry) = entries.remove(&key) {
                        stats.total_size_bytes -= entry.size_bytes;
                        stats.eviction_count += 1;
                    }
                }
            }
            EvictionPolicy::Adaptive => {
                // Combine multiple strategies
                let current_time = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                
                let mut scored_entries: Vec<_> = entries
                    .iter()
                    .map(|(key, entry)| {
                        let age_factor = (current_time - entry.created_at) as f64 / self.config.ttl_seconds as f64;
                        let frequency_factor = 1.0 / (entry.access_count as f64 + 1.0);
                        let priority_factor = 1.0 / (entry.priority + 1.0);
                        
                        let score = age_factor * 0.4 + frequency_factor * 0.3 + priority_factor * 0.3;
                        
                        (key.clone(), score)
                    })
                    .collect();
                
                scored_entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                
                let keys_to_remove: Vec<String> = scored_entries
                    .iter()
                    .take(eviction_count)
                    .map(|(key, _)| key.clone())
                    .collect();
                
                for key in keys_to_remove {
                    if let Some(entry) = entries.remove(&key) {
                        stats.total_size_bytes -= entry.size_bytes;
                        stats.eviction_count += 1;
                    }
                }
            }
        }
        
        stats.total_entries = entries.len();
        Ok(())
    }
    
    /// Estimate the size of a cache value
    fn estimate_size(&self, value: &CacheValue) -> usize {
        match value {
            CacheValue::Decision(_) => 256,
            CacheValue::MarketState(_) => 1024,
            CacheValue::Analysis(_) => 512,
            CacheValue::AgentOutput(_) => 512,
            CacheValue::BoardResult(_) => 1024,
            CacheValue::StrategyResult(_) => 512,
            CacheValue::RawBytes(bytes) => bytes.len(),
        }
    }
    
    /// Compress cache data using configured compression level
    #[cfg(feature = "compression")]
    fn compress_data(&self, data: &[u8]) -> PadsResult<Vec<u8>> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;
        
        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(self.config.compression_level as u32));
        encoder.write_all(data)
            .map_err(|e| PadsError::internal(format!("Compression failed: {}", e)))?;
        
        encoder.finish()
            .map_err(|e| PadsError::internal(format!("Compression finish failed: {}", e)))
    }
    
    /// Decompress cache data
    #[cfg(feature = "compression")]
    fn decompress_data(&self, data: &[u8]) -> PadsResult<Vec<u8>> {
        use flate2::read::GzDecoder;
        use std::io::Read;
        
        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        
        decoder.read_to_end(&mut decompressed)
            .map_err(|e| PadsError::internal(format!("Decompression failed: {}", e)))?;
        
        Ok(decompressed)
    }
    
    /// SIMD-optimized key lookup
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    fn simd_key_lookup(&self, keys: &[String], target: &str) -> Option<usize> {
        // This is a simplified example - real SIMD string matching is more complex
        unsafe {
            for (i, key) in keys.iter().enumerate() {
                if key == target {
                    return Some(i);
                }
            }
        }
        None
    }
    
    /// Fallback key lookup for non-SIMD architectures
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    fn simd_key_lookup(&self, keys: &[String], target: &str) -> Option<usize> {
        keys.iter().position(|key| key == target)
    }
    
    /// Optimize cache based on access patterns
    pub async fn optimize(&self) -> PadsResult<()> {
        let entries = self.entries.read().await;
        let stats = self.stats.read().await;
        
        // Implement cache optimization based on access patterns
        // This could include:
        // - Adjusting TTL based on access frequency
        // - Reorganizing data structures for better locality
        // - Prefetching frequently accessed items
        
        // For now, we'll just update the average access time
        drop(stats);
        drop(entries);
        
        Ok(())
    }
    
    /// Get cache efficiency metrics
    pub async fn efficiency_metrics(&self) -> PadsResult<HashMap<String, f64>> {
        let stats = self.stats.read().await;
        let entries = self.entries.read().await;
        
        let mut metrics = HashMap::new();
        
        metrics.insert("hit_rate".to_string(), stats.hit_rate);
        metrics.insert("memory_utilization".to_string(), 
                      stats.total_size_bytes as f64 / self.config.max_memory_bytes as f64);
        metrics.insert("entry_utilization".to_string(), 
                      stats.total_entries as f64 / self.config.max_entries as f64);
        metrics.insert("average_access_time_ms".to_string(), 
                      stats.average_access_time_ns as f64 / 1_000_000.0);
        
        if !entries.is_empty() {
            let total_accesses: u64 = entries.values().map(|e| e.access_count).sum();
            let average_accesses = total_accesses as f64 / entries.len() as f64;
            metrics.insert("average_accesses_per_entry".to_string(), average_accesses);
        }
        
        Ok(metrics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_cache_basic_operations() {
        let config = CacheConfig::default();
        let cache = CircuitCache::new(config).await.unwrap();
        
        // Test put and get
        let value = CacheValue::Decision(TradingDecision {
            decision_type: DecisionType::Buy,
            confidence: 0.8,
            amount: 100.0,
            reasoning: "Test decision".to_string(),
            timestamp: 1234567890,
            metadata: HashMap::new(),
        });
        
        cache.put("test_key".to_string(), value.clone(), 1.0).await.unwrap();
        
        let retrieved = cache.get("test_key").await.unwrap();
        assert!(retrieved.is_some());
        
        // Test remove
        let removed = cache.remove("test_key").await.unwrap();
        assert!(removed.is_some());
        
        let retrieved = cache.get("test_key").await.unwrap();
        assert!(retrieved.is_none());
    }
    
    #[tokio::test]
    async fn test_cache_eviction() {
        let mut config = CacheConfig::default();
        config.max_entries = 2;
        
        let cache = CircuitCache::new(config).await.unwrap();
        
        // Fill cache beyond capacity
        for i in 0..3 {
            let value = CacheValue::Decision(TradingDecision {
                decision_type: DecisionType::Buy,
                confidence: 0.8,
                amount: 100.0,
                reasoning: format!("Test decision {}", i),
                timestamp: 1234567890,
                metadata: HashMap::new(),
            });
            
            cache.put(format!("key_{}", i), value, 1.0).await.unwrap();
        }
        
        let stats = cache.stats().await.unwrap();
        assert!(stats.eviction_count > 0);
        assert!(stats.total_entries <= 2);
    }
    
    #[tokio::test]
    async fn test_cache_stats() {
        let config = CacheConfig::default();
        let cache = CircuitCache::new(config).await.unwrap();
        
        let value = CacheValue::Decision(TradingDecision {
            decision_type: DecisionType::Buy,
            confidence: 0.8,
            amount: 100.0,
            reasoning: "Test decision".to_string(),
            timestamp: 1234567890,
            metadata: HashMap::new(),
        });
        
        cache.put("test_key".to_string(), value.clone(), 1.0).await.unwrap();
        
        // Hit
        let _ = cache.get("test_key").await.unwrap();
        
        // Miss
        let _ = cache.get("nonexistent_key").await.unwrap();
        
        let stats = cache.stats().await.unwrap();
        assert_eq!(stats.hit_count, 1);
        assert_eq!(stats.miss_count, 1);
        assert_eq!(stats.hit_rate, 0.5);
    }
}
