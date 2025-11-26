# AgentDB Query Optimization Guide for Rust

**Version:** 1.0.0
**Target:** Neural Trading Rust Port
**Date:** 2025-11-12

## Table of Contents

1. [Query Performance Overview](#query-performance-overview)
2. [HNSW Index Tuning](#hnsw-index-tuning)
3. [Query Patterns](#query-patterns)
4. [Caching Strategies](#caching-strategies)
5. [Batch Operations](#batch-operations)
6. [Connection Pooling](#connection-pooling)
7. [Monitoring & Profiling](#monitoring--profiling)

---

## Query Performance Overview

### Performance Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│  Optimization Level    │ Technique       │ Speedup      │
├────────────────────────┼─────────────────┼──────────────┤
│  L0: Architecture      │ Vector-only     │ 150x         │
│  L1: Indexing          │ HNSW tuning     │ 10-50x       │
│  L2: Caching           │ Hot cache       │ 100-1000x    │
│  L3: Batching          │ Batch queries   │ 5-10x        │
│  L4: Pooling           │ Connection pool │ 2-5x         │
│  L5: Hardware          │ SSD/NVMe        │ 2-10x        │
└─────────────────────────────────────────────────────────┘
```

### Latency Budget Breakdown

For a target of <1ms vector search:

```rust
// Latency budget for vector search
const TARGET_LATENCY_US: u64 = 1000;

// Breakdown:
// - Network RTT:        200μs (20%)
// - Deserialization:    100μs (10%)
// - HNSW traversal:     500μs (50%)
// - Filtering:          150μs (15%)
// - Serialization:       50μs (5%)
```

---

## HNSW Index Tuning

### Parameter Selection

HNSW has three key parameters:

1. **M**: Number of bidirectional links per node
2. **ef_construction**: Search depth during index build
3. **ef_search**: Search depth during query

```rust
use agentdb::{VectorDBConfig, HNSWParams, IndexType};

// Decision matrix for HNSW parameters
pub fn optimal_hnsw_params(
    dataset_size: usize,
    dimension: usize,
    query_latency_target_us: u64,
) -> HNSWParams {
    match (dataset_size, query_latency_target_us) {
        // Small dataset (<10K), ultra-low latency
        (0..=10_000, 0..=500) => HNSWParams {
            m: 8,
            ef_construction: 100,
            ef_search: 30,
        },

        // Small dataset, low latency
        (0..=10_000, 501..=1000) => HNSWParams {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
        },

        // Medium dataset (10K-1M), ultra-low latency
        (10_001..=1_000_000, 0..=500) => HNSWParams {
            m: 16,
            ef_construction: 200,
            ef_search: 40,
        },

        // Medium dataset, low latency
        (10_001..=1_000_000, 501..=1000) => HNSWParams {
            m: 32,
            ef_construction: 400,
            ef_search: 100,
        },

        // Large dataset (>1M), ultra-low latency (challenging)
        (1_000_001.., 0..=500) => HNSWParams {
            m: 32,
            ef_construction: 400,
            ef_search: 50,
        },

        // Large dataset, low latency
        (1_000_001.., 501..=1000) => HNSWParams {
            m: 48,
            ef_construction: 600,
            ef_search: 150,
        },

        // Default fallback
        _ => HNSWParams {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
        },
    }
}
```

### Adaptive Parameter Tuning

```rust
use std::time::{Duration, Instant};

pub struct AdaptiveHNSW {
    current_ef_search: usize,
    target_latency: Duration,
    latency_samples: Vec<Duration>,
}

impl AdaptiveHNSW {
    pub fn new(target_latency: Duration) -> Self {
        Self {
            current_ef_search: 50,
            target_latency,
            latency_samples: Vec::with_capacity(100),
        }
    }

    /// Adjust ef_search based on observed latency
    pub fn adapt(&mut self, observed_latency: Duration) {
        self.latency_samples.push(observed_latency);

        // Adapt every 100 queries
        if self.latency_samples.len() >= 100 {
            let p95_latency = self.percentile(95.0);

            if p95_latency > self.target_latency * 2 {
                // Too slow, decrease ef_search
                self.current_ef_search = (self.current_ef_search * 8 / 10).max(10);
                println!("Decreasing ef_search to {}", self.current_ef_search);
            } else if p95_latency < self.target_latency / 2 {
                // Too fast, increase ef_search for better recall
                self.current_ef_search = (self.current_ef_search * 12 / 10).min(500);
                println!("Increasing ef_search to {}", self.current_ef_search);
            }

            self.latency_samples.clear();
        }
    }

    fn percentile(&self, p: f64) -> Duration {
        let mut sorted = self.latency_samples.clone();
        sorted.sort();
        let idx = ((sorted.len() as f64) * p / 100.0) as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    pub fn current_ef(&self) -> usize {
        self.current_ef_search
    }
}
```

### Recall vs Latency Tradeoff

```rust
pub struct RecallLatencyProfile {
    ef_search: usize,
    recall: f64,
    latency_us: u64,
}

/// Benchmark different ef_search values
pub async fn benchmark_ef_search(
    db: &VectorDB,
    query_vectors: Vec<Vec<f32>>,
    ground_truth: Vec<Vec<Uuid>>,
) -> Vec<RecallLatencyProfile> {
    let ef_values = vec![10, 20, 30, 50, 100, 200, 500];
    let mut profiles = Vec::new();

    for ef in ef_values {
        let mut total_latency = Duration::ZERO;
        let mut total_recall = 0.0;

        for (query, truth) in query_vectors.iter().zip(&ground_truth) {
            let start = Instant::now();
            let results = db
                .search(Query::new(query).k(10).ef_search(ef))
                .await
                .unwrap();
            total_latency += start.elapsed();

            // Calculate recall@10
            let result_ids: HashSet<_> = results.iter().map(|r| r.id).collect();
            let truth_ids: HashSet<_> = truth.iter().cloned().collect();
            let recall = result_ids.intersection(&truth_ids).count() as f64 / truth.len() as f64;
            total_recall += recall;
        }

        let avg_latency = total_latency / query_vectors.len() as u32;
        let avg_recall = total_recall / query_vectors.len() as f64;

        profiles.push(RecallLatencyProfile {
            ef_search: ef,
            recall: avg_recall,
            latency_us: avg_latency.as_micros() as u64,
        });

        println!(
            "ef_search={}: recall={:.3}, latency={}μs",
            ef, avg_recall, avg_latency.as_micros()
        );
    }

    profiles
}
```

---

## Query Patterns

### 1. Point Query (Fastest)

```rust
/// Retrieve single item by ID (hash lookup)
pub async fn get_by_id(
    db: &VectorDB,
    id: Uuid,
) -> Result<Observation, AgentDBError> {
    // Direct hash lookup: O(1), <100μs
    db.get(id.as_bytes()).await
}
```

**Performance:** <100μs, no HNSW traversal

### 2. KNN Query (Fast)

```rust
/// K-nearest neighbors search
pub async fn knn_search(
    db: &VectorDB,
    query_vec: &[f32],
    k: usize,
) -> Result<Vec<Observation>, AgentDBError> {
    // HNSW traversal: O(log N), <1ms
    db.search(Query::new(query_vec).k(k)).await
}
```

**Performance:** <1ms for k=10, <2ms for k=100

### 3. Filtered KNN (Medium)

```rust
/// KNN with metadata filter
pub async fn filtered_knn(
    db: &VectorDB,
    query_vec: &[f32],
    k: usize,
    symbol: &str,
) -> Result<Vec<Observation>, AgentDBError> {
    // HNSW + post-filtering: O(log N + F), <2ms
    db.search(
        Query::new(query_vec)
            .k(k)
            .filter(Filter::eq("symbol", symbol))
    ).await
}
```

**Performance:** <2ms

**Optimization:** Use pre-filtered index if filter is common:

```rust
/// Create separate index per symbol for frequent queries
pub struct SymbolIndexes {
    indexes: HashMap<String, VectorDB>,
}

impl SymbolIndexes {
    pub async fn get_or_create(&mut self, symbol: &str) -> &VectorDB {
        self.indexes
            .entry(symbol.to_string())
            .or_insert_with(|| {
                VectorDBConfig::new()
                    .dimension(512)
                    .index_type(IndexType::HNSW)
                    .build()
                    .unwrap()
            })
    }

    pub async fn search(
        &self,
        symbol: &str,
        query_vec: &[f32],
        k: usize,
    ) -> Result<Vec<Observation>, AgentDBError> {
        // Direct HNSW search without filter: <1ms
        self.indexes
            .get(symbol)
            .ok_or(AgentDBError::IndexNotFound)?
            .search(Query::new(query_vec).k(k))
            .await
    }
}
```

### 4. Range Query (Slow)

```rust
/// Temporal range query
pub async fn range_query(
    db: &VectorDB,
    symbol: &str,
    start_us: i64,
    end_us: i64,
) -> Result<Vec<Observation>, AgentDBError> {
    // Full scan with filter: O(N), <50ms for 1M items
    db.search(
        Query::new_filter(
            Filter::and(vec![
                Filter::eq("symbol", symbol),
                Filter::gte("timestamp_us", start_us),
                Filter::lte("timestamp_us", end_us),
            ])
        ).limit(10000)
    ).await
}
```

**Performance:** <50ms for 1-day window, <500ms for 1-month window

**Optimization:** Use time-partitioned indexes:

```rust
pub struct TimePartitionedIndex {
    partitions: BTreeMap<i64, VectorDB>,
    partition_duration_us: i64,
}

impl TimePartitionedIndex {
    pub fn new(partition_duration: Duration) -> Self {
        Self {
            partitions: BTreeMap::new(),
            partition_duration_us: partition_duration.as_micros() as i64,
        }
    }

    fn partition_key(&self, timestamp_us: i64) -> i64 {
        timestamp_us / self.partition_duration_us
    }

    pub async fn insert(&mut self, obs: &Observation) -> Result<(), AgentDBError> {
        let key = self.partition_key(obs.timestamp_us);
        let partition = self.partitions
            .entry(key)
            .or_insert_with(|| self.create_partition());

        partition.insert(obs.id.as_bytes(), &obs.embedding, Some(obs)).await
    }

    pub async fn range_query(
        &self,
        start_us: i64,
        end_us: i64,
    ) -> Result<Vec<Observation>, AgentDBError> {
        let start_key = self.partition_key(start_us);
        let end_key = self.partition_key(end_us);

        let mut results = Vec::new();

        // Only scan relevant partitions: 10-100x faster
        for (_, partition) in self.partitions.range(start_key..=end_key) {
            let obs = partition
                .search(
                    Query::new_filter(
                        Filter::and(vec![
                            Filter::gte("timestamp_us", start_us),
                            Filter::lte("timestamp_us", end_us),
                        ])
                    ).limit(10000)
                ).await?;

            results.extend(obs);
        }

        Ok(results)
    }
}
```

---

## Caching Strategies

### 1. Query Result Cache

```rust
use lru::LruCache;
use std::sync::Mutex;
use blake3::Hasher;

pub struct QueryCache<T> {
    cache: Mutex<LruCache<u64, Vec<T>>>,
}

impl<T: Clone> QueryCache<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: Mutex::new(LruCache::new(capacity)),
        }
    }

    /// Generate cache key from query
    fn query_key(&self, query_vec: &[f32], k: usize, filter: Option<&str>) -> u64 {
        let mut hasher = Hasher::new();

        // Hash query vector
        for &v in query_vec {
            hasher.update(&v.to_le_bytes());
        }

        // Hash parameters
        hasher.update(&k.to_le_bytes());

        if let Some(f) = filter {
            hasher.update(f.as_bytes());
        }

        let hash = hasher.finalize();
        u64::from_le_bytes(hash.as_bytes()[0..8].try_into().unwrap())
    }

    pub fn get(&self, query_vec: &[f32], k: usize, filter: Option<&str>) -> Option<Vec<T>> {
        let key = self.query_key(query_vec, k, filter);
        self.cache.lock().unwrap().get(&key).cloned()
    }

    pub fn put(&self, query_vec: &[f32], k: usize, filter: Option<&str>, results: Vec<T>) {
        let key = self.query_key(query_vec, k, filter);
        self.cache.lock().unwrap().put(key, results);
    }
}

/// Usage with cache
pub async fn cached_knn_search(
    db: &VectorDB,
    cache: &QueryCache<Observation>,
    query_vec: &[f32],
    k: usize,
) -> Result<Vec<Observation>, AgentDBError> {
    // Check cache first: <1μs
    if let Some(cached) = cache.get(query_vec, k, None) {
        return Ok(cached);
    }

    // Cache miss, query database: <1ms
    let results = db.search(Query::new(query_vec).k(k)).await?;

    // Store in cache
    cache.put(query_vec, k, None, results.clone());

    Ok(results)
}
```

### 2. Embedding Cache

```rust
use std::collections::HashMap;
use std::sync::RwLock;

pub struct EmbeddingCache {
    cache: RwLock<HashMap<String, Vec<f32>>>,
}

impl EmbeddingCache {
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
        }
    }

    pub fn get(&self, key: &str) -> Option<Vec<f32>> {
        self.cache.read().unwrap().get(key).cloned()
    }

    pub fn put(&self, key: String, embedding: Vec<f32>) {
        self.cache.write().unwrap().insert(key, embedding);
    }

    /// Get or compute embedding
    pub fn get_or_compute<F>(&self, key: &str, compute: F) -> Vec<f32>
    where
        F: FnOnce() -> Vec<f32>,
    {
        // Fast path: read lock
        if let Some(emb) = self.get(key) {
            return emb;
        }

        // Slow path: compute and write
        let emb = compute();
        self.put(key.to_string(), emb.clone());
        emb
    }
}
```

### 3. Hot Data Cache

```rust
use std::time::{Duration, Instant};

pub struct HotDataCache<T> {
    data: RwLock<HashMap<Uuid, (T, Instant)>>,
    ttl: Duration,
}

impl<T: Clone> HotDataCache<T> {
    pub fn new(ttl: Duration) -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
            ttl,
        }
    }

    pub fn get(&self, id: &Uuid) -> Option<T> {
        let cache = self.data.read().unwrap();
        if let Some((value, inserted)) = cache.get(id) {
            if inserted.elapsed() < self.ttl {
                return Some(value.clone());
            }
        }
        None
    }

    pub fn put(&self, id: Uuid, value: T) {
        self.data.write().unwrap().insert(id, (value, Instant::now()));
    }

    /// Evict expired entries
    pub fn evict_expired(&self) {
        let mut cache = self.data.write().unwrap();
        cache.retain(|_, (_, inserted)| inserted.elapsed() < self.ttl);
    }
}
```

---

## Batch Operations

### Batch Insert

```rust
/// Batch insert with optimal throughput
pub async fn batch_insert_observations(
    db: &VectorDB,
    observations: Vec<Observation>,
    batch_size: usize,
) -> Result<(), AgentDBError> {
    // Process in chunks
    for chunk in observations.chunks(batch_size) {
        let batch: Vec<_> = chunk
            .iter()
            .map(|obs| {
                (
                    obs.id.as_bytes(),
                    obs.embedding.as_slice(),
                    Some(serde_json::to_value(obs).unwrap()),
                )
            })
            .collect();

        db.batch_insert(batch).await?;
    }

    Ok(())
}
```

**Optimal batch size:** 1000-5000 items

### Batch Query

```rust
/// Batch multiple queries in parallel
pub async fn batch_knn_queries(
    db: &VectorDB,
    queries: Vec<Vec<f32>>,
    k: usize,
) -> Result<Vec<Vec<Observation>>, AgentDBError> {
    // Build queries
    let batch_queries: Vec<_> = queries
        .iter()
        .map(|vec| Query::new(vec).k(k))
        .collect();

    // Execute in parallel
    db.batch_search(batch_queries).await
}
```

### Pipelined Operations

```rust
use tokio::sync::mpsc;

pub struct Pipeline<T, R> {
    input_tx: mpsc::Sender<T>,
    output_rx: mpsc::Receiver<R>,
}

impl<T, R> Pipeline<T, R>
where
    T: Send + 'static,
    R: Send + 'static,
{
    pub fn new<F, Fut>(workers: usize, process: F) -> Self
    where
        F: Fn(T) -> Fut + Send + Clone + 'static,
        Fut: std::future::Future<Output = R> + Send,
    {
        let (input_tx, mut input_rx) = mpsc::channel(1000);
        let (output_tx, output_rx) = mpsc::channel(1000);

        // Spawn worker tasks
        for _ in 0..workers {
            let mut rx = input_rx.clone();
            let tx = output_tx.clone();
            let process = process.clone();

            tokio::spawn(async move {
                while let Some(item) = rx.recv().await {
                    let result = process(item).await;
                    let _ = tx.send(result).await;
                }
            });
        }

        Self { input_tx, output_rx }
    }

    pub async fn send(&self, item: T) -> Result<(), mpsc::error::SendError<T>> {
        self.input_tx.send(item).await
    }

    pub async fn recv(&mut self) -> Option<R> {
        self.output_rx.recv().await
    }
}

/// Usage: Parallel observation processing
pub async fn parallel_store_observations(
    db: Arc<VectorDB>,
    observations: Vec<Observation>,
) -> Result<(), AgentDBError> {
    let pipeline = Pipeline::new(8, move |obs: Observation| {
        let db = db.clone();
        async move {
            db.insert(obs.id.as_bytes(), &obs.embedding, Some(&obs))
                .await
                .unwrap();
        }
    });

    // Send all observations
    for obs in observations {
        pipeline.send(obs).await.unwrap();
    }

    Ok(())
}
```

---

## Connection Pooling

```rust
use deadpool::managed::{Manager, Pool, RecycleResult};
use async_trait::async_trait;

pub struct AgentDBManager {
    endpoint: String,
}

#[async_trait]
impl Manager for AgentDBManager {
    type Type = AgentDBClient;
    type Error = AgentDBError;

    async fn create(&self) -> Result<AgentDBClient, AgentDBError> {
        AgentDBClient::connect(&self.endpoint).await
    }

    async fn recycle(&self, conn: &mut AgentDBClient) -> RecycleResult<AgentDBError> {
        // Check if connection is still alive
        match conn.ping().await {
            Ok(_) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }
}

pub type AgentDBPool = Pool<AgentDBManager>;

pub async fn create_pool(endpoint: String, size: usize) -> Result<AgentDBPool, AgentDBError> {
    let manager = AgentDBManager { endpoint };
    Pool::builder(manager)
        .max_size(size)
        .build()
        .map_err(|e| AgentDBError::PoolError(e.to_string()))
}

/// Usage with pool
pub async fn query_with_pool(
    pool: &AgentDBPool,
    query: Query,
) -> Result<Vec<Observation>, AgentDBError> {
    let client = pool.get().await?;
    client.search(query).await
}
```

---

## Monitoring & Profiling

### Query Metrics

```rust
use prometheus::{
    Counter, Histogram, IntCounter, Registry,
    opts, histogram_opts,
};

pub struct QueryMetrics {
    /// Total queries executed
    pub queries_total: IntCounter,

    /// Query latency histogram
    pub query_duration: Histogram,

    /// Cache hit rate
    pub cache_hits: IntCounter,
    pub cache_misses: IntCounter,

    /// Errors
    pub errors_total: Counter,
}

impl QueryMetrics {
    pub fn new(registry: &Registry) -> Result<Self, prometheus::Error> {
        let queries_total = IntCounter::new(
            "agentdb_queries_total",
            "Total number of queries"
        )?;
        registry.register(Box::new(queries_total.clone()))?;

        let query_duration = Histogram::with_opts(
            histogram_opts!(
                "agentdb_query_duration_seconds",
                "Query latency in seconds",
                vec![0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1]
            )
        )?;
        registry.register(Box::new(query_duration.clone()))?;

        let cache_hits = IntCounter::new(
            "agentdb_cache_hits_total",
            "Total cache hits"
        )?;
        registry.register(Box::new(cache_hits.clone()))?;

        let cache_misses = IntCounter::new(
            "agentdb_cache_misses_total",
            "Total cache misses"
        )?;
        registry.register(Box::new(cache_misses.clone()))?;

        let errors_total = Counter::new(
            "agentdb_errors_total",
            "Total errors"
        )?;
        registry.register(Box::new(errors_total.clone()))?;

        Ok(Self {
            queries_total,
            query_duration,
            cache_hits,
            cache_misses,
            errors_total,
        })
    }

    pub fn record_query(&self, duration: Duration) {
        self.queries_total.inc();
        self.query_duration.observe(duration.as_secs_f64());
    }

    pub fn record_cache_hit(&self) {
        self.cache_hits.inc();
    }

    pub fn record_cache_miss(&self) {
        self.cache_misses.inc();
    }

    pub fn record_error(&self) {
        self.errors_total.inc();
    }

    pub fn cache_hit_rate(&self) -> f64 {
        let hits = self.cache_hits.get() as f64;
        let total = hits + self.cache_misses.get() as f64;
        if total == 0.0 {
            0.0
        } else {
            hits / total
        }
    }
}
```

### Query Profiler

```rust
use std::sync::Arc;
use tracing::{info, warn};

pub struct QueryProfiler {
    metrics: Arc<QueryMetrics>,
    cache: Arc<QueryCache<Observation>>,
}

impl QueryProfiler {
    pub fn new(metrics: Arc<QueryMetrics>, cache: Arc<QueryCache<Observation>>) -> Self {
        Self { metrics, cache }
    }

    pub async fn profile_query<F, Fut, T>(
        &self,
        name: &str,
        f: F,
    ) -> Result<T, AgentDBError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T, AgentDBError>>,
    {
        let start = Instant::now();

        let result = f().await;

        let duration = start.elapsed();
        self.metrics.record_query(duration);

        match &result {
            Ok(_) => {
                info!(
                    query = name,
                    duration_us = duration.as_micros(),
                    "Query succeeded"
                );
            }
            Err(e) => {
                self.metrics.record_error();
                warn!(
                    query = name,
                    duration_us = duration.as_micros(),
                    error = ?e,
                    "Query failed"
                );
            }
        }

        result
    }
}
```

---

## Performance Checklist

### Pre-Production

- [ ] Benchmark HNSW parameters for your dataset
- [ ] Implement query result caching
- [ ] Set up connection pooling
- [ ] Enable time partitioning for range queries
- [ ] Create separate indexes for common filters
- [ ] Monitor cache hit rates
- [ ] Profile query latencies
- [ ] Load test with realistic traffic

### Production

- [ ] Monitor p95 and p99 latencies
- [ ] Set up alerting for slow queries (>10ms)
- [ ] Track cache hit rate (target >80%)
- [ ] Monitor connection pool saturation
- [ ] Profile periodically for regressions
- [ ] Optimize hottest queries first
- [ ] Consider read replicas for high load
- [ ] Implement query timeouts

---

## Troubleshooting

### Query Too Slow (>10ms)

1. **Check ef_search**: Lower for faster queries
2. **Check filters**: Remove or pre-filter with separate index
3. **Check dataset size**: Consider sharding
4. **Check M parameter**: Increase for better connectivity
5. **Check cache**: Ensure caching is enabled

### Low Cache Hit Rate (<50%)

1. **Check cache size**: Increase capacity
2. **Check query diversity**: High diversity = low hit rate
3. **Check TTL**: Increase for stable data
4. **Check cache key**: Ensure deterministic

### High Memory Usage

1. **Enable quantization**: 4-32x reduction
2. **Reduce M parameter**: Lower memory footprint
3. **Reduce cache size**: Trade latency for memory
4. **Partition data**: Multiple smaller indexes

---

## Conclusion

Query optimization is crucial for sub-millisecond performance. Focus on:

1. **HNSW tuning** for your dataset characteristics
2. **Caching** for frequently accessed data
3. **Batching** for throughput
4. **Monitoring** for continuous improvement

Target: <1ms for 95% of queries, <2ms for 99% of queries.

---

**Next:** Implement and benchmark your queries
**Status:** Guide Complete ✅
