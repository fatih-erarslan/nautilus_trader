//! # Cache Management Agent
//!
//! Intelligent caching and memory optimization agent for high-frequency data processing.
//! Provides advanced caching strategies with memory-optimized data structures.

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::{HashMap, BTreeMap};
use tokio::sync::{RwLock, mpsc, Mutex};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error, debug};
use async_trait::async_trait;
use lru::LruCache;
use std::num::NonZeroUsize;

use crate::agents::base::{
    DataAgent, DataAgentId, DataAgentType, DataAgentState, DataAgentInfo,
    DataMessage, DataMessageType, MessageMetadata, MessagePriority,
    CoordinationMessage, HealthStatus, HealthLevel, HealthMetrics,
    AgentMetrics, BaseDataAgent, MetricsUpdate
};

/// Cache management agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheManagementConfig {
    /// Target cache access latency in microseconds
    pub target_latency_us: u64,
    /// Cache strategies configuration
    pub cache_strategies: CacheStrategiesConfig,
    /// Memory management configuration
    pub memory_config: MemoryManagementConfig,
    /// Eviction policies configuration
    pub eviction_config: EvictionConfig,
    /// Compression settings
    pub compression_config: CompressionConfig,
    /// Distributed caching settings
    pub distributed_config: DistributedCacheConfig,
}

impl Default for CacheManagementConfig {
    fn default() -> Self {
        Self {
            target_latency_us: 10,
            cache_strategies: CacheStrategiesConfig::default(),
            memory_config: MemoryManagementConfig::default(),
            eviction_config: EvictionConfig::default(),
            compression_config: CompressionConfig::default(),
            distributed_config: DistributedCacheConfig::default(),
        }
    }
}

/// Cache strategies configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStrategiesConfig {
    /// Enabled cache strategies
    pub strategies: Vec<CacheStrategy>,
    /// Primary cache strategy
    pub primary_strategy: CacheStrategy,
    /// Cache hierarchy
    pub cache_hierarchy: Vec<CacheLevel>,
    /// Adaptive caching settings
    pub adaptive_caching: AdaptiveCachingConfig,
}

impl Default for CacheStrategiesConfig {
    fn default() -> Self {
        Self {
            strategies: vec![
                CacheStrategy::LRU,
                CacheStrategy::LFU,
                CacheStrategy::ARC,
            ],
            primary_strategy: CacheStrategy::ARC,
            cache_hierarchy: vec![
                CacheLevel::L1,
                CacheLevel::L2,
                CacheLevel::L3,
            ],
            adaptive_caching: AdaptiveCachingConfig::default(),
        }
    }
}

/// Cache strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CacheStrategy {
    LRU,
    LFU,
    FIFO,
    Random,
    ARC,
    CAR,
    Clock,
    TwoQ,
}

/// Cache levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CacheLevel {
    L1,
    L2,
    L3,
    Distributed,
}

/// Adaptive caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveCachingConfig {
    /// Enable adaptive caching
    pub enabled: bool,
    /// Learning algorithm
    pub learning_algorithm: LearningAlgorithm,
    /// Adaptation interval
    pub adaptation_interval: Duration,
    /// Performance threshold
    pub performance_threshold: f64,
    /// Strategy switching enabled
    pub strategy_switching: bool,
}

impl Default for AdaptiveCachingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            learning_algorithm: LearningAlgorithm::ReinforcementLearning,
            adaptation_interval: Duration::from_secs(60),
            performance_threshold: 0.8,
            strategy_switching: true,
        }
    }
}

/// Learning algorithms for adaptive caching
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LearningAlgorithm {
    ReinforcementLearning,
    MachineLearning,
    HeuristicBased,
    StatisticalAnalysis,
}

/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryManagementConfig {
    /// Total cache size in MB
    pub total_cache_size_mb: usize,
    /// L1 cache size in MB
    pub l1_cache_size_mb: usize,
    /// L2 cache size in MB
    pub l2_cache_size_mb: usize,
    /// L3 cache size in MB
    pub l3_cache_size_mb: usize,
    /// Memory allocation strategy
    pub allocation_strategy: AllocationStrategy,
    /// Memory optimization settings
    pub optimization_settings: MemoryOptimizationSettings,
}

impl Default for MemoryManagementConfig {
    fn default() -> Self {
        Self {
            total_cache_size_mb: 2048,
            l1_cache_size_mb: 64,
            l2_cache_size_mb: 256,
            l3_cache_size_mb: 1024,
            allocation_strategy: AllocationStrategy::Adaptive,
            optimization_settings: MemoryOptimizationSettings::default(),
        }
    }
}

/// Memory allocation strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AllocationStrategy {
    Static,
    Dynamic,
    Adaptive,
    Predictive,
}

/// Memory optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimizationSettings {
    /// Enable memory pooling
    pub memory_pooling: bool,
    /// Enable object pooling
    pub object_pooling: bool,
    /// Enable memory mapping
    pub memory_mapping: bool,
    /// Enable huge pages
    pub huge_pages: bool,
    /// Memory alignment
    pub memory_alignment: usize,
    /// Prefetch distance
    pub prefetch_distance: usize,
}

impl Default for MemoryOptimizationSettings {
    fn default() -> Self {
        Self {
            memory_pooling: true,
            object_pooling: true,
            memory_mapping: true,
            huge_pages: true,
            memory_alignment: 64,
            prefetch_distance: 4,
        }
    }
}

/// Eviction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvictionConfig {
    /// Eviction policies
    pub policies: Vec<EvictionPolicy>,
    /// Primary eviction policy
    pub primary_policy: EvictionPolicy,
    /// Eviction thresholds
    pub eviction_thresholds: EvictionThresholds,
    /// Background eviction settings
    pub background_eviction: BackgroundEvictionConfig,
}

impl Default for EvictionConfig {
    fn default() -> Self {
        Self {
            policies: vec![
                EvictionPolicy::LRU,
                EvictionPolicy::LFU,
                EvictionPolicy::TTL,
            ],
            primary_policy: EvictionPolicy::LRU,
            eviction_thresholds: EvictionThresholds::default(),
            background_eviction: BackgroundEvictionConfig::default(),
        }
    }
}

/// Eviction policies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    FIFO,
    Random,
    TTL,
    Size,
    Priority,
    Hybrid,
}

/// Eviction thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvictionThresholds {
    /// Memory usage threshold (0.0 - 1.0)
    pub memory_threshold: f64,
    /// Cache hit rate threshold
    pub hit_rate_threshold: f64,
    /// Maximum item age
    pub max_item_age: Duration,
    /// Maximum item size
    pub max_item_size_mb: usize,
}

impl Default for EvictionThresholds {
    fn default() -> Self {
        Self {
            memory_threshold: 0.9,
            hit_rate_threshold: 0.7,
            max_item_age: Duration::from_secs(3600),
            max_item_size_mb: 10,
        }
    }
}

/// Background eviction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundEvictionConfig {
    /// Enable background eviction
    pub enabled: bool,
    /// Eviction interval
    pub eviction_interval: Duration,
    /// Batch size for eviction
    pub batch_size: usize,
    /// Maximum eviction time
    pub max_eviction_time: Duration,
}

impl Default for BackgroundEvictionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            eviction_interval: Duration::from_secs(30),
            batch_size: 100,
            max_eviction_time: Duration::from_millis(10),
        }
    }
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Enable compression
    pub enabled: bool,
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level
    pub level: u8,
    /// Minimum size for compression
    pub min_compression_size: usize,
    /// Compression ratio threshold
    pub compression_ratio_threshold: f64,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: CompressionAlgorithm::LZ4,
            level: 1,
            min_compression_size: 1024,
            compression_ratio_threshold: 0.8,
        }
    }
}

/// Compression algorithms
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    LZ4,
    Zstd,
    Snappy,
    Gzip,
    Brotli,
}

/// Distributed cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedCacheConfig {
    /// Enable distributed caching
    pub enabled: bool,
    /// Cache nodes
    pub nodes: Vec<CacheNode>,
    /// Consistency level
    pub consistency_level: ConsistencyLevel,
    /// Replication factor
    pub replication_factor: usize,
    /// Partitioning strategy
    pub partitioning_strategy: PartitioningStrategy,
}

impl Default for DistributedCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            nodes: Vec::new(),
            consistency_level: ConsistencyLevel::Eventual,
            replication_factor: 2,
            partitioning_strategy: PartitioningStrategy::ConsistentHashing,
        }
    }
}

/// Cache node configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheNode {
    pub id: String,
    pub address: String,
    pub port: u16,
    pub weight: f64,
    pub status: NodeStatus,
}

/// Node status
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NodeStatus {
    Online,
    Offline,
    Degraded,
    Maintenance,
}

/// Consistency levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    Strong,
    Eventual,
    Weak,
    Causal,
}

/// Partitioning strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PartitioningStrategy {
    ConsistentHashing,
    RangePartitioning,
    HashPartitioning,
    RandomPartitioning,
}

/// Cache item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheItem {
    pub key: String,
    pub value: serde_json::Value,
    pub metadata: CacheItemMetadata,
}

/// Cache item metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheItemMetadata {
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_accessed: chrono::DateTime<chrono::Utc>,
    pub access_count: u64,
    pub size_bytes: usize,
    pub ttl: Option<Duration>,
    pub priority: CachePriority,
    pub compressed: bool,
    pub compression_ratio: f64,
}

/// Cache priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CachePriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Intelligent cache
pub struct IntelligentCache {
    config: Arc<CacheManagementConfig>,
    l1_cache: Arc<RwLock<LruCache<String, CacheItem>>>,
    l2_cache: Arc<RwLock<LruCache<String, CacheItem>>>,
    l3_cache: Arc<RwLock<LruCache<String, CacheItem>>>,
    access_patterns: Arc<RwLock<HashMap<String, AccessPattern>>>,
    performance_metrics: Arc<RwLock<CachePerformanceMetrics>>,
}

/// Access pattern tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessPattern {
    pub key: String,
    pub access_frequency: f64,
    pub access_recency: chrono::DateTime<chrono::Utc>,
    pub access_history: Vec<chrono::DateTime<chrono::Utc>>,
    pub predicted_next_access: Option<chrono::DateTime<chrono::Utc>>,
}

/// Cache performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachePerformanceMetrics {
    pub hit_rate: f64,
    pub miss_rate: f64,
    pub eviction_rate: f64,
    pub average_access_time_us: f64,
    pub memory_utilization: f64,
    pub compression_ratio: f64,
    pub cache_efficiency: f64,
}

impl Default for CachePerformanceMetrics {
    fn default() -> Self {
        Self {
            hit_rate: 0.0,
            miss_rate: 0.0,
            eviction_rate: 0.0,
            average_access_time_us: 0.0,
            memory_utilization: 0.0,
            compression_ratio: 1.0,
            cache_efficiency: 0.0,
        }
    }
}

impl IntelligentCache {
    /// Create a new intelligent cache
    pub fn new(config: Arc<CacheManagementConfig>) -> Self {
        let l1_size = NonZeroUsize::new(config.memory_config.l1_cache_size_mb * 1024).unwrap();
        let l2_size = NonZeroUsize::new(config.memory_config.l2_cache_size_mb * 1024).unwrap();
        let l3_size = NonZeroUsize::new(config.memory_config.l3_cache_size_mb * 1024).unwrap();
        
        Self {
            config,
            l1_cache: Arc::new(RwLock::new(LruCache::new(l1_size))),
            l2_cache: Arc::new(RwLock::new(LruCache::new(l2_size))),
            l3_cache: Arc::new(RwLock::new(LruCache::new(l3_size))),
            access_patterns: Arc::new(RwLock::new(HashMap::new())),
            performance_metrics: Arc::new(RwLock::new(CachePerformanceMetrics::default())),
        }
    }
    
    /// Get item from cache
    pub async fn get(&self, key: &str) -> Option<CacheItem> {
        let start_time = Instant::now();
        
        // Try L1 cache first
        if let Some(mut item) = self.l1_cache.write().await.get_mut(key) {
            item.metadata.last_accessed = chrono::Utc::now();
            item.metadata.access_count += 1;
            self.update_access_pattern(key).await;
            self.update_performance_metrics(true, start_time.elapsed()).await;
            return Some(item.clone());
        }
        
        // Try L2 cache
        if let Some(item) = self.l2_cache.write().await.pop(key) {
            // Promote to L1
            self.l1_cache.write().await.put(key.to_string(), item.clone());
            self.update_access_pattern(key).await;
            self.update_performance_metrics(true, start_time.elapsed()).await;
            return Some(item);
        }
        
        // Try L3 cache
        if let Some(item) = self.l3_cache.write().await.pop(key) {
            // Promote to L1
            self.l1_cache.write().await.put(key.to_string(), item.clone());
            self.update_access_pattern(key).await;
            self.update_performance_metrics(true, start_time.elapsed()).await;
            return Some(item);
        }
        
        // Cache miss
        self.update_performance_metrics(false, start_time.elapsed()).await;
        None
    }
    
    /// Put item in cache
    pub async fn put(&self, key: String, value: serde_json::Value, priority: CachePriority) -> Result<()> {
        let now = chrono::Utc::now();
        let value_size = value.to_string().len();
        
        // Compress if needed
        let (compressed_value, compressed, compression_ratio) = if self.should_compress(&value) {
            let compressed = self.compress_value(&value)?;
            let ratio = compressed.len() as f64 / value_size as f64;
            (serde_json::to_value(compressed)?, true, ratio)
        } else {
            (value, false, 1.0)
        };
        
        let item = CacheItem {
            key: key.clone(),
            value: compressed_value,
            metadata: CacheItemMetadata {
                created_at: now,
                last_accessed: now,
                access_count: 0,
                size_bytes: value_size,
                ttl: None,
                priority,
                compressed,
                compression_ratio,
            },
        };
        
        // Determine which cache level to use based on priority and size
        match priority {
            CachePriority::Critical | CachePriority::High => {
                self.l1_cache.write().await.put(key.clone(), item);
            }
            CachePriority::Normal => {
                self.l2_cache.write().await.put(key.clone(), item);
            }
            CachePriority::Low => {
                self.l3_cache.write().await.put(key.clone(), item);
            }
        }
        
        self.update_access_pattern(&key).await;
        Ok(())
    }
    
    /// Remove item from cache
    pub async fn remove(&self, key: &str) -> Option<CacheItem> {
        // Try all cache levels
        if let Some(item) = self.l1_cache.write().await.pop(key) {
            return Some(item);
        }
        
        if let Some(item) = self.l2_cache.write().await.pop(key) {
            return Some(item);
        }
        
        if let Some(item) = self.l3_cache.write().await.pop(key) {
            return Some(item);
        }
        
        None
    }
    
    /// Check if value should be compressed
    fn should_compress(&self, value: &serde_json::Value) -> bool {
        if !self.config.compression_config.enabled {
            return false;
        }
        
        let size = value.to_string().len();
        size >= self.config.compression_config.min_compression_size
    }
    
    /// Compress value
    fn compress_value(&self, value: &serde_json::Value) -> Result<Vec<u8>> {
        let data = value.to_string().into_bytes();
        
        match self.config.compression_config.algorithm {
            CompressionAlgorithm::LZ4 => {
                Ok(lz4::compress(&data, None, false)?)
            }
            CompressionAlgorithm::Zstd => {
                Ok(zstd::encode_all(&data[..], self.config.compression_config.level as i32)?)
            }
            _ => {
                // Other compression algorithms would be implemented here
                Ok(data)
            }
        }
    }
    
    /// Update access pattern
    async fn update_access_pattern(&self, key: &str) {
        let mut patterns = self.access_patterns.write().await;
        let pattern = patterns.entry(key.to_string()).or_insert_with(|| AccessPattern {
            key: key.to_string(),
            access_frequency: 0.0,
            access_recency: chrono::Utc::now(),
            access_history: Vec::new(),
            predicted_next_access: None,
        });
        
        let now = chrono::Utc::now();
        pattern.access_recency = now;
        pattern.access_history.push(now);
        pattern.access_frequency += 1.0;
        
        // Keep only recent history
        pattern.access_history.retain(|&time| {
            now.signed_duration_since(time).num_hours() < 24
        });
        
        // Predict next access (simplified)
        if pattern.access_history.len() > 2 {
            let intervals: Vec<_> = pattern.access_history.windows(2)
                .map(|w| w[1].signed_duration_since(w[0]).num_seconds())
                .collect();
            
            if !intervals.is_empty() {
                let avg_interval = intervals.iter().sum::<i64>() / intervals.len() as i64;
                pattern.predicted_next_access = Some(now + chrono::Duration::seconds(avg_interval));
            }
        }
    }
    
    /// Update performance metrics
    async fn update_performance_metrics(&self, hit: bool, access_time: Duration) {
        let mut metrics = self.performance_metrics.write().await;
        
        if hit {
            metrics.hit_rate = (metrics.hit_rate + 1.0) / 2.0;
            metrics.miss_rate = 1.0 - metrics.hit_rate;
        } else {
            metrics.miss_rate = (metrics.miss_rate + 1.0) / 2.0;
            metrics.hit_rate = 1.0 - metrics.miss_rate;
        }
        
        let access_time_us = access_time.as_micros() as f64;
        metrics.average_access_time_us = (metrics.average_access_time_us + access_time_us) / 2.0;
        
        // Calculate cache efficiency
        metrics.cache_efficiency = metrics.hit_rate * (1.0 / (1.0 + metrics.average_access_time_us / 1000.0));
    }
    
    /// Get cache statistics
    pub async fn get_statistics(&self) -> CacheStatistics {
        let l1_len = self.l1_cache.read().await.len();
        let l2_len = self.l2_cache.read().await.len();
        let l3_len = self.l3_cache.read().await.len();
        let performance = self.performance_metrics.read().await.clone();
        
        CacheStatistics {
            l1_items: l1_len,
            l2_items: l2_len,
            l3_items: l3_len,
            total_items: l1_len + l2_len + l3_len,
            hit_rate: performance.hit_rate,
            miss_rate: performance.miss_rate,
            average_access_time_us: performance.average_access_time_us,
            memory_utilization: performance.memory_utilization,
            compression_ratio: performance.compression_ratio,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    pub l1_items: usize,
    pub l2_items: usize,
    pub l3_items: usize,
    pub total_items: usize,
    pub hit_rate: f64,
    pub miss_rate: f64,
    pub average_access_time_us: f64,
    pub memory_utilization: f64,
    pub compression_ratio: f64,
}

/// Cache management agent
pub struct CacheManagementAgent {
    base: BaseDataAgent,
    config: Arc<CacheManagementConfig>,
    cache: Arc<IntelligentCache>,
    cache_metrics: Arc<RwLock<CacheMetrics>>,
    state: Arc<RwLock<CacheState>>,
}

/// Cache metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    pub cache_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub evictions: u64,
    pub average_response_time_us: f64,
    pub memory_usage_mb: f64,
    pub compression_savings_mb: f64,
    pub last_update: chrono::DateTime<chrono::Utc>,
}

impl Default for CacheMetrics {
    fn default() -> Self {
        Self {
            cache_requests: 0,
            cache_hits: 0,
            cache_misses: 0,
            evictions: 0,
            average_response_time_us: 0.0,
            memory_usage_mb: 0.0,
            compression_savings_mb: 0.0,
            last_update: chrono::Utc::now(),
        }
    }
}

/// Cache state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheState {
    pub active_caches: usize,
    pub total_cache_size_mb: f64,
    pub cache_utilization: f64,
    pub eviction_pressure: f64,
    pub is_healthy: bool,
    pub last_health_check: chrono::DateTime<chrono::Utc>,
}

impl Default for CacheState {
    fn default() -> Self {
        Self {
            active_caches: 3, // L1, L2, L3
            total_cache_size_mb: 0.0,
            cache_utilization: 0.0,
            eviction_pressure: 0.0,
            is_healthy: true,
            last_health_check: chrono::Utc::now(),
        }
    }
}

impl CacheManagementAgent {
    /// Create a new cache management agent
    pub async fn new(config: CacheManagementConfig) -> Result<Self> {
        let base = BaseDataAgent::new(DataAgentType::CacheManagement);
        let config = Arc::new(config);
        let cache = Arc::new(IntelligentCache::new(config.clone()));
        let cache_metrics = Arc::new(RwLock::new(CacheMetrics::default()));
        let state = Arc::new(RwLock::new(CacheState::default()));
        
        Ok(Self {
            base,
            config,
            cache,
            cache_metrics,
            state,
        })
    }
    
    /// Cache data
    pub async fn cache_data(&self, key: String, value: serde_json::Value, priority: CachePriority) -> Result<()> {
        let start_time = Instant::now();
        
        self.cache.put(key, value, priority).await?;
        
        // Update metrics
        let response_time = start_time.elapsed().as_micros() as f64;
        {
            let mut metrics = self.cache_metrics.write().await;
            metrics.cache_requests += 1;
            metrics.average_response_time_us = 
                (metrics.average_response_time_us + response_time) / 2.0;
            metrics.last_update = chrono::Utc::now();
        }
        
        Ok(())
    }
    
    /// Retrieve cached data
    pub async fn get_cached_data(&self, key: &str) -> Option<serde_json::Value> {
        let start_time = Instant::now();
        
        let result = self.cache.get(key).await;
        
        // Update metrics
        let response_time = start_time.elapsed().as_micros() as f64;
        {
            let mut metrics = self.cache_metrics.write().await;
            metrics.cache_requests += 1;
            
            if result.is_some() {
                metrics.cache_hits += 1;
            } else {
                metrics.cache_misses += 1;
            }
            
            metrics.average_response_time_us = 
                (metrics.average_response_time_us + response_time) / 2.0;
            metrics.last_update = chrono::Utc::now();
        }
        
        result.map(|item| item.value)
    }
    
    /// Remove cached data
    pub async fn remove_cached_data(&self, key: &str) -> Option<serde_json::Value> {
        self.cache.remove(key).await.map(|item| item.value)
    }
    
    /// Get cache metrics
    pub async fn get_cache_metrics(&self) -> CacheMetrics {
        self.cache_metrics.read().await.clone()
    }
    
    /// Get cache state
    pub async fn get_cache_state(&self) -> CacheState {
        self.state.read().await.clone()
    }
    
    /// Get cache statistics
    pub async fn get_cache_statistics(&self) -> CacheStatistics {
        self.cache.get_statistics().await
    }
}

#[async_trait]
impl DataAgent for CacheManagementAgent {
    fn get_id(&self) -> DataAgentId {
        self.base.id
    }
    
    fn get_type(&self) -> DataAgentType {
        DataAgentType::CacheManagement
    }
    
    async fn get_state(&self) -> DataAgentState {
        self.base.state.read().await.clone()
    }
    
    async fn get_info(&self) -> DataAgentInfo {
        self.base.info.read().await.clone()
    }
    
    async fn start(&self) -> Result<()> {
        info!("Starting cache management agent");
        
        self.base.update_state(DataAgentState::Running).await?;
        
        info!("Cache management agent started successfully");
        Ok(())
    }
    
    async fn stop(&self) -> Result<()> {
        info!("Stopping cache management agent");
        
        self.base.update_state(DataAgentState::Stopping).await?;
        
        // Clear caches if needed
        // Note: In a real implementation, you might want to persist cache data
        
        self.base.update_state(DataAgentState::Stopped).await?;
        
        info!("Cache management agent stopped successfully");
        Ok(())
    }
    
    async fn process(&self, message: DataMessage) -> Result<DataMessage> {
        let start_time = Instant::now();
        
        match message.message_type {
            DataMessageType::CacheRequest => {
                // Handle cache request
                let cache_key = message.payload.get("key")
                    .and_then(|k| k.as_str())
                    .unwrap_or("unknown");
                
                let cached_data = self.get_cached_data(cache_key).await;
                
                let response_payload = match cached_data {
                    Some(data) => serde_json::json!({
                        "status": "hit",
                        "data": data
                    }),
                    None => serde_json::json!({
                        "status": "miss"
                    })
                };
                
                let response = DataMessage {
                    id: uuid::Uuid::new_v4(),
                    timestamp: chrono::Utc::now(),
                    source: self.get_id(),
                    destination: message.destination,
                    message_type: DataMessageType::CacheResponse,
                    payload: response_payload,
                    metadata: MessageMetadata {
                        priority: MessagePriority::High,
                        expires_at: Some(chrono::Utc::now() + chrono::Duration::seconds(30)),
                        retry_count: 0,
                        trace_id: format!("cache_management_{}", uuid::Uuid::new_v4()),
                        span_id: format!("span_{}", uuid::Uuid::new_v4()),
                    },
                };
                
                Ok(response)
            }
            _ => {
                // Handle other message types by caching the data
                let cache_key = format!("msg_{}", message.id);
                self.cache_data(cache_key, message.payload.clone(), CachePriority::Normal).await?;
                
                let response = DataMessage {
                    id: uuid::Uuid::new_v4(),
                    timestamp: chrono::Utc::now(),
                    source: self.get_id(),
                    destination: message.destination,
                    message_type: DataMessageType::CacheResponse,
                    payload: serde_json::json!({
                        "status": "cached",
                        "cache_key": format!("msg_{}", message.id)
                    }),
                    metadata: MessageMetadata {
                        priority: MessagePriority::Normal,
                        expires_at: Some(chrono::Utc::now() + chrono::Duration::seconds(30)),
                        retry_count: 0,
                        trace_id: format!("cache_management_{}", uuid::Uuid::new_v4()),
                        span_id: format!("span_{}", uuid::Uuid::new_v4()),
                    },
                };
                
                Ok(response)
            }
        }?;
        
        // Update metrics
        let latency = start_time.elapsed().as_micros() as f64;
        self.base.update_metrics(MetricsUpdate::MessageProcessed(latency)).await?;
        
        Ok(response)
    }
    
    async fn health_check(&self) -> Result<HealthStatus> {
        let state = self.get_cache_state().await;
        let metrics = self.get_cache_metrics().await;
        let statistics = self.get_cache_statistics().await;
        
        let health_level = if state.is_healthy && statistics.hit_rate > 0.5 {
            HealthLevel::Healthy
        } else if statistics.hit_rate > 0.3 {
            HealthLevel::Warning
        } else {
            HealthLevel::Critical
        };
        
        Ok(HealthStatus {
            status: health_level,
            last_check: chrono::Utc::now(),
            uptime: self.base.start_time.elapsed(),
            issues: Vec::new(),
            metrics: HealthMetrics {
                cpu_usage_percent: 0.0, // Would be measured
                memory_usage_mb: metrics.memory_usage_mb,
                network_usage_mbps: 0.0, // Would be measured
                disk_usage_mb: 0.0, // Would be measured
                error_rate: metrics.cache_misses as f64 / metrics.cache_requests.max(1) as f64,
                response_time_ms: metrics.average_response_time_us / 1000.0,
            },
        })
    }
    
    async fn get_metrics(&self) -> Result<AgentMetrics> {
        Ok(self.base.metrics.read().await.clone())
    }
    
    async fn reset(&self) -> Result<()> {
        info!("Resetting cache management agent");
        
        // Reset metrics
        {
            let mut metrics = self.cache_metrics.write().await;
            *metrics = CacheMetrics::default();
        }
        
        // Reset state
        {
            let mut state = self.state.write().await;
            *state = CacheState::default();
        }
        
        info!("Cache management agent reset successfully");
        Ok(())
    }
    
    async fn handle_coordination(&self, message: CoordinationMessage) -> Result<()> {
        debug!("Handling coordination message: {:?}", message.coordination_type);
        
        match message.coordination_type {
            crate::agents::base::CoordinationType::LoadBalancing => {
                info!("Received load balancing coordination");
            }
            crate::agents::base::CoordinationType::StateSync => {
                info!("Received state sync coordination");
            }
            _ => {
                debug!("Unhandled coordination type: {:?}", message.coordination_type);
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;
    
    #[test]
    async fn test_cache_management_agent_creation() {
        let config = CacheManagementConfig::default();
        let agent = CacheManagementAgent::new(config).await;
        assert!(agent.is_ok());
    }
    
    #[test]
    async fn test_intelligent_cache() {
        let config = CacheManagementConfig::default();
        let cache = IntelligentCache::new(Arc::new(config));
        
        // Test cache operations
        let key = "test_key".to_string();
        let value = serde_json::json!({"data": "test_value"});
        
        let put_result = cache.put(key.clone(), value.clone(), CachePriority::Normal).await;
        assert!(put_result.is_ok());
        
        let get_result = cache.get(&key).await;
        assert!(get_result.is_some());
        assert_eq!(get_result.unwrap().value, value);
    }
    
    #[test]
    async fn test_cache_operations() {
        let config = CacheManagementConfig::default();
        let agent = CacheManagementAgent::new(config).await.unwrap();
        
        let key = "test_cache_key".to_string();
        let value = serde_json::json!({"test": "data"});
        
        // Test caching
        let cache_result = agent.cache_data(key.clone(), value.clone(), CachePriority::High).await;
        assert!(cache_result.is_ok());
        
        // Test retrieval
        let retrieved = agent.get_cached_data(&key).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), value);
        
        // Test removal
        let removed = agent.remove_cached_data(&key).await;
        assert!(removed.is_some());
        
        // Verify removal
        let not_found = agent.get_cached_data(&key).await;
        assert!(not_found.is_none());
    }
    
    #[test]
    async fn test_cache_statistics() {
        let config = CacheManagementConfig::default();
        let agent = CacheManagementAgent::new(config).await.unwrap();
        
        let stats = agent.get_cache_statistics().await;
        assert_eq!(stats.total_items, 0);
        
        // Add some items
        agent.cache_data("key1".to_string(), serde_json::json!("value1"), CachePriority::High).await.unwrap();
        agent.cache_data("key2".to_string(), serde_json::json!("value2"), CachePriority::Normal).await.unwrap();
        
        let stats_after = agent.get_cache_statistics().await;
        assert!(stats_after.total_items > 0);
    }
}