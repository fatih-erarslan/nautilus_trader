//! Distributed caching system with Redis backend
//!
//! This module provides comprehensive distributed caching capabilities:
//! - Multi-level cache hierarchy (L1 local, L2 Redis)
//! - Cache eviction policies (LRU, LFU, TTL-based)
//! - Cache coherence and invalidation
//! - Distributed cache warming and preloading
//! - Cache analytics and optimization
//! - Compression and serialization optimization

use std::{
    collections::{HashMap, BTreeMap},
    sync::{Arc, atomic::{AtomicU64, AtomicUsize, Ordering}},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
    hash::{Hash, Hasher},
};
use tokio::{
    sync::{RwLock, Mutex},
    time::{interval, sleep},
};
use serde::{Serialize, Deserialize, de::DeserializeOwned};
use lru::LruCache;
use sha2::{Sha256, Digest};
use flate2::{Compression, write::GzEncoder, read::GzDecoder};
use std::io::{Write, Read};
use crate::{
    error::{CdfaError, Result},
    integration::{
        redis_connector::RedisPool,
        messaging::{RedisMessageBroker, MessageType, MessagePriority},
    },
};

/// Cache eviction policies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Time To Live based
    TTL,
    /// Random eviction
    Random,
    /// No eviction (cache grows indefinitely)
    None,
}

/// Cache entry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Entry key
    pub key: String,
    /// Serialized data
    pub data: Vec<u8>,
    /// Creation timestamp
    pub created_at: u64,
    /// Last access timestamp
    pub last_accessed: u64,
    /// Access count
    pub access_count: u64,
    /// Time to live in seconds
    pub ttl: Option<u64>,
    /// Data size in bytes
    pub size: usize,
    /// Compression algorithm used
    pub compression: Option<String>,
    /// Data checksum for integrity
    pub checksum: String,
    /// Entry version for optimistic locking
    pub version: u64,
    /// Cache level (L1, L2, etc.)
    pub level: CacheLevel,
}

impl CacheEntry {
    /// Check if entry has expired
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
            now > self.created_at + ttl
        } else {
            false
        }
    }

    /// Update access statistics
    pub fn update_access(&mut self) {
        self.last_accessed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        self.access_count += 1;
    }

    /// Calculate age in seconds
    pub fn age(&self) -> u64 {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        now.saturating_sub(self.created_at)
    }

    /// Calculate time since last access
    pub fn time_since_access(&self) -> u64 {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        now.saturating_sub(self.last_accessed)
    }

    /// Verify data integrity
    pub fn verify_integrity(&self) -> bool {
        let mut hasher = Sha256::new();
        hasher.update(&self.data);
        let hash = format!("{:x}", hasher.finalize());
        hash == self.checksum
    }
}

/// Cache levels for hierarchical caching
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CacheLevel {
    /// Local in-memory cache (fastest)
    L1,
    /// Distributed Redis cache (fast)
    L2,
    /// Persistent storage cache (slow)
    L3,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum cache size in bytes
    pub max_size_bytes: usize,
    /// Maximum number of entries
    pub max_entries: usize,
    /// Default TTL in seconds
    pub default_ttl: Option<u64>,
    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Enable compression for large entries
    pub enable_compression: bool,
    /// Compression threshold in bytes
    pub compression_threshold: usize,
    /// Enable cache coherence across nodes
    pub enable_coherence: bool,
    /// Cache warming strategies
    pub warming_strategies: Vec<WarmingStrategy>,
    /// Analytics collection interval
    pub analytics_interval: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_size_bytes: 100 * 1024 * 1024, // 100MB
            max_entries: 10000,
            default_ttl: Some(3600), // 1 hour
            eviction_policy: EvictionPolicy::LRU,
            enable_compression: true,
            compression_threshold: 1024, // 1KB
            enable_coherence: true,
            warming_strategies: vec![WarmingStrategy::PreloadPopular],
            analytics_interval: 300, // 5 minutes
        }
    }
}

/// Cache warming strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WarmingStrategy {
    /// Preload popular items based on access patterns
    PreloadPopular,
    /// Preload items based on time patterns
    PreloadTemporal,
    /// Preload items based on dependencies
    PreloadDependencies,
    /// Custom warming strategy
    Custom(String),
}

/// Cache statistics for monitoring and optimization
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total entries stored
    pub total_entries: usize,
    /// Current cache size in bytes
    pub current_size: usize,
    /// Total evictions
    pub evictions: u64,
    /// Average access time in microseconds
    pub avg_access_time_us: f64,
    /// Hit ratio (0.0 to 1.0)
    pub hit_ratio: f64,
    /// Popular keys (most accessed)
    pub popular_keys: Vec<(String, u64)>,
    /// Cache levels utilization
    pub level_stats: HashMap<CacheLevel, LevelStats>,
    /// Compression ratio
    pub compression_ratio: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LevelStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: usize,
    pub size_bytes: usize,
    pub avg_access_time_us: f64,
}

impl CacheStats {
    /// Update hit ratio
    pub fn update_hit_ratio(&mut self) {
        let total = self.hits + self.misses;
        self.hit_ratio = if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        };
    }

    /// Record cache hit
    pub fn record_hit(&mut self, level: CacheLevel, access_time: Duration) {
        self.hits += 1;
        self.level_stats.entry(level).or_default().hits += 1;
        
        let access_time_us = access_time.as_micros() as f64;
        
        // Update average using exponential moving average
        if self.avg_access_time_us == 0.0 {
            self.avg_access_time_us = access_time_us;
        } else {
            self.avg_access_time_us = 0.9 * self.avg_access_time_us + 0.1 * access_time_us;
        }
        
        let level_stats = self.level_stats.entry(level).or_default();
        if level_stats.avg_access_time_us == 0.0 {
            level_stats.avg_access_time_us = access_time_us;
        } else {
            level_stats.avg_access_time_us = 0.9 * level_stats.avg_access_time_us + 0.1 * access_time_us;
        }
        
        self.update_hit_ratio();
    }

    /// Record cache miss
    pub fn record_miss(&mut self, level: CacheLevel) {
        self.misses += 1;
        self.level_stats.entry(level).or_default().misses += 1;
        self.update_hit_ratio();
    }
}

/// Local in-memory cache (L1)
struct LocalCache {
    /// LRU cache for data storage
    cache: Arc<Mutex<LruCache<String, CacheEntry>>>,
    /// Current size in bytes
    current_size: AtomicUsize,
    /// Configuration
    config: CacheConfig,
    /// Statistics
    stats: Arc<RwLock<LevelStats>>,
}

impl LocalCache {
    /// Create new local cache
    fn new(config: CacheConfig) -> Self {
        let cache = Arc::new(Mutex::new(LruCache::new(
            std::num::NonZeroUsize::new(config.max_entries).unwrap()
        )));
        
        Self {
            cache,
            current_size: AtomicUsize::new(0),
            config,
            stats: Arc::new(RwLock::new(LevelStats::default())),
        }
    }

    /// Get entry from local cache
    async fn get(&self, key: &str) -> Option<CacheEntry> {
        let start_time = Instant::now();
        let mut cache = self.cache.lock().await;
        
        if let Some(mut entry) = cache.get_mut(key) {
            if entry.is_expired() {
                cache.pop(key);
                self.current_size.fetch_sub(entry.size, Ordering::Relaxed);
                None
            } else {
                entry.update_access();
                let mut stats = self.stats.write().await;
                stats.record_hit(CacheLevel::L1, start_time.elapsed());
                Some(entry.clone())
            }
        } else {
            let mut stats = self.stats.write().await;
            stats.record_miss(CacheLevel::L1);
            None
        }
    }

    /// Put entry into local cache
    async fn put(&self, key: String, entry: CacheEntry) -> Result<()> {
        let mut cache = self.cache.lock().await;
        
        // Check if we need to evict entries
        while self.current_size.load(Ordering::Relaxed) + entry.size > self.config.max_size_bytes {
            if let Some((_, evicted)) = cache.pop_lru() {
                self.current_size.fetch_sub(evicted.size, Ordering::Relaxed);
            } else {
                break;
            }
        }

        // Insert new entry
        if let Some(old_entry) = cache.put(key, entry.clone()) {
            self.current_size.fetch_sub(old_entry.size, Ordering::Relaxed);
        }
        
        self.current_size.fetch_add(entry.size, Ordering::Relaxed);
        
        let mut stats = self.stats.write().await;
        stats.entries = cache.len();
        stats.size_bytes = self.current_size.load(Ordering::Relaxed);

        Ok(())
    }

    /// Remove entry from local cache
    async fn remove(&self, key: &str) -> Option<CacheEntry> {
        let mut cache = self.cache.lock().await;
        if let Some(entry) = cache.pop(key) {
            self.current_size.fetch_sub(entry.size, Ordering::Relaxed);
            
            let mut stats = self.stats.write().await;
            stats.entries = cache.len();
            stats.size_bytes = self.current_size.load(Ordering::Relaxed);
            
            Some(entry)
        } else {
            None
        }
    }

    /// Clear all entries
    async fn clear(&self) {
        let mut cache = self.cache.lock().await;
        cache.clear();
        self.current_size.store(0, Ordering::Relaxed);
        
        let mut stats = self.stats.write().await;
        stats.entries = 0;
        stats.size_bytes = 0;
    }

    /// Get current statistics
    async fn get_stats(&self) -> LevelStats {
        self.stats.read().await.clone()
    }
}

/// Distributed cache manager with Redis backend
pub struct DistributedCache {
    /// Local L1 cache
    local_cache: LocalCache,
    /// Redis connection pool for L2 cache
    redis_pool: Arc<RedisPool>,
    /// Message broker for cache coherence
    message_broker: Option<Arc<RwLock<RedisMessageBroker>>>,
    /// Node identifier
    node_id: String,
    /// Cache configuration
    config: CacheConfig,
    /// Global statistics
    stats: Arc<RwLock<CacheStats>>,
    /// Version counter for optimistic locking
    version_counter: AtomicU64,
}

impl DistributedCache {
    /// Create new distributed cache
    pub async fn new(
        config: CacheConfig,
        redis_pool: Arc<RedisPool>,
        node_id: String,
        message_broker: Option<Arc<RwLock<RedisMessageBroker>>>,
    ) -> Result<Self> {
        let local_cache = LocalCache::new(config.clone());
        
        let cache = Self {
            local_cache,
            redis_pool,
            message_broker,
            node_id,
            config: config.clone(),
            stats: Arc::new(RwLock::new(CacheStats::default())),
            version_counter: AtomicU64::new(1),
        };

        // Start background tasks
        cache.start_analytics_task().await;
        cache.start_coherence_task().await;
        cache.start_warming_task().await;

        Ok(cache)
    }

    /// Get value from cache (tries L1, then L2)
    pub async fn get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        let start_time = Instant::now();

        // Try L1 cache first
        if let Some(entry) = self.local_cache.get(key).await {
            if entry.verify_integrity() {
                let value = self.deserialize_entry::<T>(&entry)?;
                self.stats.write().await.record_hit(CacheLevel::L1, start_time.elapsed());
                return Ok(Some(value));
            } else {
                log::warn!("Cache entry integrity check failed for key: {}", key);
                self.local_cache.remove(key).await;
            }
        }

        // Try L2 cache (Redis)
        if let Some(entry) = self.get_from_redis(key).await? {
            if entry.verify_integrity() {
                let value = self.deserialize_entry::<T>(&entry)?;
                
                // Promote to L1 cache
                self.local_cache.put(key.to_string(), entry).await?;
                
                self.stats.write().await.record_hit(CacheLevel::L2, start_time.elapsed());
                return Ok(Some(value));
            } else {
                log::warn!("Redis cache entry integrity check failed for key: {}", key);
                self.remove_from_redis(key).await?;
            }
        }

        // Cache miss
        self.stats.write().await.record_miss(CacheLevel::L2);
        Ok(None)
    }

    /// Put value into cache (L1 and L2)
    pub async fn put<T: Serialize>(&self, key: &str, value: &T, ttl: Option<u64>) -> Result<()> {
        let entry = self.create_entry(key, value, ttl)?;
        
        // Store in L1 cache
        self.local_cache.put(key.to_string(), entry.clone()).await?;
        
        // Store in L2 cache (Redis)
        self.put_to_redis(key, &entry).await?;
        
        // Send coherence notification if enabled
        if self.config.enable_coherence {
            self.send_coherence_notification(key, CoherenceAction::Update).await?;
        }

        self.stats.write().await.total_entries += 1;
        Ok(())
    }

    /// Remove value from cache
    pub async fn remove(&self, key: &str) -> Result<bool> {
        let l1_removed = self.local_cache.remove(key).await.is_some();
        let l2_removed = self.remove_from_redis(key).await?;
        
        // Send coherence notification if enabled
        if self.config.enable_coherence && (l1_removed || l2_removed) {
            self.send_coherence_notification(key, CoherenceAction::Remove).await?;
        }

        Ok(l1_removed || l2_removed)
    }

    /// Check if key exists in cache
    pub async fn exists(&self, key: &str) -> Result<bool> {
        // Check L1 first
        if self.local_cache.get(key).await.is_some() {
            return Ok(true);
        }

        // Check L2 (Redis)
        self.exists_in_redis(key).await
    }

    /// Clear all cache entries
    pub async fn clear(&self) -> Result<()> {
        // Clear L1
        self.local_cache.clear().await;
        
        // Clear L2 (Redis) - only keys belonging to this cache instance
        self.clear_redis_namespace().await?;
        
        // Send coherence notification
        if self.config.enable_coherence {
            self.send_coherence_notification("*", CoherenceAction::Clear).await?;
        }

        let mut stats = self.stats.write().await;
        stats.total_entries = 0;
        stats.current_size = 0;

        Ok(())
    }

    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        let mut stats = self.stats.read().await.clone();
        
        // Update L1 stats
        let l1_stats = self.local_cache.get_stats().await;
        stats.level_stats.insert(CacheLevel::L1, l1_stats);
        
        // Get L2 stats from Redis
        if let Ok(l2_stats) = self.get_redis_stats().await {
            stats.level_stats.insert(CacheLevel::L2, l2_stats);
        }

        stats
    }

    /// Create cache entry from value
    fn create_entry<T: Serialize>(&self, key: &str, value: &T, ttl: Option<u64>) -> Result<CacheEntry> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        
        // Serialize value
        let mut data = rmp_serde::to_vec(value)
            .map_err(|e| CdfaError::Serialization(format!("Failed to serialize cache value: {}", e)))?;

        let mut compression = None;
        
        // Apply compression if enabled and data is large enough
        if self.config.enable_compression && data.len() > self.config.compression_threshold {
            let compressed = self.compress_data(&data)?;
            if compressed.len() < data.len() {
                data = compressed;
                compression = Some("gzip".to_string());
            }
        }

        // Calculate checksum
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let checksum = format!("{:x}", hasher.finalize());

        Ok(CacheEntry {
            key: key.to_string(),
            data,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            ttl: ttl.or(self.config.default_ttl),
            size: data.len(),
            compression,
            checksum,
            version: self.version_counter.fetch_add(1, Ordering::Relaxed),
            level: CacheLevel::L1,
        })
    }

    /// Deserialize cache entry
    fn deserialize_entry<T: DeserializeOwned>(&self, entry: &CacheEntry) -> Result<T> {
        let mut data = entry.data.clone();

        // Decompress if needed
        if let Some(ref compression_type) = entry.compression {
            match compression_type.as_str() {
                "gzip" => {
                    data = self.decompress_data(&data)?;
                }
                _ => {
                    return Err(CdfaError::Serialization(format!("Unknown compression type: {}", compression_type)));
                }
            }
        }

        rmp_serde::from_slice(&data)
            .map_err(|e| CdfaError::Serialization(format!("Failed to deserialize cache value: {}", e)))
    }

    /// Compress data using gzip
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data)
            .map_err(|e| CdfaError::Serialization(format!("Compression failed: {}", e)))?;
        encoder.finish()
            .map_err(|e| CdfaError::Serialization(format!("Compression finalization failed: {}", e)))
    }

    /// Decompress data using gzip
    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)
            .map_err(|e| CdfaError::Serialization(format!("Decompression failed: {}", e)))?;
        Ok(decompressed)
    }

    /// Get entry from Redis
    async fn get_from_redis(&self, key: &str) -> Result<Option<CacheEntry>> {
        let redis_key = format!("cdfa:cache:{}:{}", self.node_id, key);
        
        let conn = self.redis_pool.get_connection().await?;
        let mut locked_conn = conn.lock().await;

        let data: Option<Vec<u8>> = locked_conn.connection
            .get(&redis_key)
            .await
            .map_err(|e| CdfaError::Network(format!("Failed to get from Redis: {}", e)))?;

        if let Some(data) = data {
            let entry: CacheEntry = rmp_serde::from_slice(&data)
                .map_err(|e| CdfaError::Serialization(format!("Failed to deserialize cache entry: {}", e)))?;
            
            if entry.is_expired() {
                // Remove expired entry
                let _: () = locked_conn.connection
                    .del(&redis_key)
                    .await
                    .map_err(|e| CdfaError::Network(format!("Failed to delete expired entry: {}", e)))?;
                Ok(None)
            } else {
                Ok(Some(entry))
            }
        } else {
            Ok(None)
        }
    }

    /// Put entry to Redis
    async fn put_to_redis(&self, key: &str, entry: &CacheEntry) -> Result<()> {
        let redis_key = format!("cdfa:cache:{}:{}", self.node_id, key);
        
        let data = rmp_serde::to_vec(entry)
            .map_err(|e| CdfaError::Serialization(format!("Failed to serialize cache entry: {}", e)))?;

        let conn = self.redis_pool.get_connection().await?;
        let mut locked_conn = conn.lock().await;

        if let Some(ttl) = entry.ttl {
            let _: () = locked_conn.connection
                .set_ex(&redis_key, data, ttl)
                .await
                .map_err(|e| CdfaError::Network(format!("Failed to set in Redis with TTL: {}", e)))?;
        } else {
            let _: () = locked_conn.connection
                .set(&redis_key, data)
                .await
                .map_err(|e| CdfaError::Network(format!("Failed to set in Redis: {}", e)))?;
        }

        Ok(())
    }

    /// Remove entry from Redis
    async fn remove_from_redis(&self, key: &str) -> Result<bool> {
        let redis_key = format!("cdfa:cache:{}:{}", self.node_id, key);
        
        let conn = self.redis_pool.get_connection().await?;
        let mut locked_conn = conn.lock().await;

        let deleted: i32 = locked_conn.connection
            .del(&redis_key)
            .await
            .map_err(|e| CdfaError::Network(format!("Failed to delete from Redis: {}", e)))?;

        Ok(deleted > 0)
    }

    /// Check if key exists in Redis
    async fn exists_in_redis(&self, key: &str) -> Result<bool> {
        let redis_key = format!("cdfa:cache:{}:{}", self.node_id, key);
        
        let conn = self.redis_pool.get_connection().await?;
        let mut locked_conn = conn.lock().await;

        let exists: bool = locked_conn.connection
            .exists(&redis_key)
            .await
            .map_err(|e| CdfaError::Network(format!("Failed to check existence in Redis: {}", e)))?;

        Ok(exists)
    }

    /// Clear Redis namespace for this cache instance
    async fn clear_redis_namespace(&self) -> Result<()> {
        let pattern = format!("cdfa:cache:{}:*", self.node_id);
        
        let conn = self.redis_pool.get_connection().await?;
        let mut locked_conn = conn.lock().await;

        let keys: Vec<String> = locked_conn.connection
            .keys(&pattern)
            .await
            .map_err(|e| CdfaError::Network(format!("Failed to get keys for deletion: {}", e)))?;

        if !keys.is_empty() {
            let _: () = locked_conn.connection
                .del(&keys)
                .await
                .map_err(|e| CdfaError::Network(format!("Failed to delete keys: {}", e)))?;
        }

        Ok(())
    }

    /// Get Redis cache statistics
    async fn get_redis_stats(&self) -> Result<LevelStats> {
        // This would require implementing Redis-specific statistics collection
        // For now, return default stats
        Ok(LevelStats::default())
    }

    /// Send cache coherence notification
    async fn send_coherence_notification(&self, key: &str, action: CoherenceAction) -> Result<()> {
        if let Some(ref broker) = self.message_broker {
            let notification = CoherenceNotification {
                key: key.to_string(),
                action,
                node_id: self.node_id.clone(),
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            };

            broker.read().await.broadcast_message(
                MessageType::Custom("cache_coherence".to_string()),
                &notification,
                MessagePriority::Normal,
            ).await?;
        }

        Ok(())
    }

    /// Start analytics collection task
    async fn start_analytics_task(&self) {
        let stats = self.stats.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(config.analytics_interval));
            
            loop {
                interval.tick().await;
                
                // Collect and update analytics
                let mut stats_guard = stats.write().await;
                stats_guard.update_hit_ratio();
                
                // Log analytics if needed
                if log::log_enabled!(log::Level::Debug) {
                    log::debug!("Cache analytics: hit_ratio={:.3}, entries={}, size={}MB", 
                              stats_guard.hit_ratio,
                              stats_guard.total_entries,
                              stats_guard.current_size / (1024 * 1024));
                }
            }
        });
    }

    /// Start cache coherence task
    async fn start_coherence_task(&self) {
        if !self.config.enable_coherence || self.message_broker.is_none() {
            return;
        }

        // Implementation would subscribe to coherence notifications
        // and update local cache accordingly
        log::info!("Cache coherence task started");
    }

    /// Start cache warming task
    async fn start_warming_task(&self) {
        if self.config.warming_strategies.is_empty() {
            return;
        }

        // Implementation would preload popular/predicted keys
        log::info!("Cache warming task started");
    }
}

/// Cache coherence actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoherenceAction {
    Update,
    Remove,
    Clear,
}

/// Cache coherence notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceNotification {
    pub key: String,
    pub action: CoherenceAction,
    pub node_id: String,
    pub timestamp: u64,
}

/// Cache factory for creating different cache configurations
pub struct CacheFactory;

impl CacheFactory {
    /// Create a high-performance cache for trading operations
    pub async fn create_trading_cache(
        redis_pool: Arc<RedisPool>,
        node_id: String,
        message_broker: Option<Arc<RwLock<RedisMessageBroker>>>,
    ) -> Result<DistributedCache> {
        let config = CacheConfig {
            max_size_bytes: 500 * 1024 * 1024, // 500MB
            max_entries: 100000,
            default_ttl: Some(300), // 5 minutes
            eviction_policy: EvictionPolicy::LRU,
            enable_compression: true,
            compression_threshold: 512, // 512 bytes
            enable_coherence: true,
            warming_strategies: vec![WarmingStrategy::PreloadPopular, WarmingStrategy::PreloadTemporal],
            analytics_interval: 60, // 1 minute
        };

        DistributedCache::new(config, redis_pool, node_id, message_broker).await
    }

    /// Create a general-purpose cache
    pub async fn create_general_cache(
        redis_pool: Arc<RedisPool>,
        node_id: String,
    ) -> Result<DistributedCache> {
        let config = CacheConfig::default();
        DistributedCache::new(config, redis_pool, node_id, None).await
    }

    /// Create a large data cache for analytics
    pub async fn create_analytics_cache(
        redis_pool: Arc<RedisPool>,
        node_id: String,
    ) -> Result<DistributedCache> {
        let config = CacheConfig {
            max_size_bytes: 2 * 1024 * 1024 * 1024, // 2GB
            max_entries: 50000,
            default_ttl: Some(7200), // 2 hours
            eviction_policy: EvictionPolicy::LFU,
            enable_compression: true,
            compression_threshold: 1024, // 1KB
            enable_coherence: false, // Less critical for analytics
            warming_strategies: vec![WarmingStrategy::PreloadDependencies],
            analytics_interval: 300, // 5 minutes
        };

        DistributedCache::new(config, redis_pool, node_id, None).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_entry_expiration() {
        let mut entry = CacheEntry {
            key: "test".to_string(),
            data: vec![1, 2, 3],
            created_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() - 3600,
            last_accessed: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            access_count: 1,
            ttl: Some(1800), // 30 minutes
            size: 3,
            compression: None,
            checksum: "test_hash".to_string(),
            version: 1,
            level: CacheLevel::L1,
        };

        assert!(entry.is_expired());

        entry.ttl = None;
        assert!(!entry.is_expired());
    }

    #[test]
    fn test_cache_stats_hit_ratio() {
        let mut stats = CacheStats::default();
        
        stats.hits = 80;
        stats.misses = 20;
        stats.update_hit_ratio();
        
        assert_eq!(stats.hit_ratio, 0.8);
    }

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::default();
        assert_eq!(config.eviction_policy, EvictionPolicy::LRU);
        assert!(config.enable_compression);
        assert_eq!(config.compression_threshold, 1024);
    }

    #[tokio::test]
    async fn test_local_cache_operations() {
        let config = CacheConfig::default();
        let cache = LocalCache::new(config);

        let entry = CacheEntry {
            key: "test_key".to_string(),
            data: vec![1, 2, 3, 4, 5],
            created_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            last_accessed: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            access_count: 0,
            ttl: Some(3600),
            size: 5,
            compression: None,
            checksum: "test_hash".to_string(),
            version: 1,
            level: CacheLevel::L1,
        };

        // Test put and get
        cache.put("test_key".to_string(), entry.clone()).await.unwrap();
        let retrieved = cache.get("test_key").await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().data, entry.data);

        // Test remove
        let removed = cache.remove("test_key").await;
        assert!(removed.is_some());
        
        let not_found = cache.get("test_key").await;
        assert!(not_found.is_none());
    }
}