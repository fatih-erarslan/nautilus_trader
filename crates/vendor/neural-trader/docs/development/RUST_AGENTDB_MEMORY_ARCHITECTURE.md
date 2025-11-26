# Neural Trading Rust Port - AgentDB Memory Architecture

**Version:** 1.0.0
**Target Performance:** Sub-microsecond query latency, 150x faster than SQL
**Status:** Design Document
**Date:** 2025-11-12

## Table of Contents

1. [Overview](#overview)
2. [Architecture Principles](#architecture-principles)
3. [Schema Design](#schema-design)
4. [Memory Patterns](#memory-patterns)
5. [AgentDB Integration](#agentdb-integration)
6. [Query Patterns](#query-patterns)
7. [Provenance & Cryptography](#provenance--cryptography)
8. [Performance Targets](#performance-targets)
9. [Rust Implementation](#rust-implementation)
10. [Benchmarking Strategy](#benchmarking-strategy)

---

## Overview

This document defines the memory architecture and AgentDB integration for the Neural Trading Rust port. The design prioritizes:

- **Ultra-low latency**: Sub-microsecond query performance for trading decisions
- **Vector-first design**: Hash-based embeddings for deterministic lookups
- **ReasoningBank integration**: Reflection loops for continuous learning
- **Provenance tracking**: Cryptographic signatures for audit trails
- **Zero-copy optimization**: Minimize allocations in hot paths

### Key Metrics

| Metric | Target | Current (Python) | Improvement |
|--------|--------|------------------|-------------|
| Vector Search | <1ms | 150-300ms | 150-300x |
| Cache Hit Rate | >85% | 0-75% | New capability |
| Memory Footprint | -32x | Baseline | Via quantization |
| Decision Latency | <100μs | 5-50ms | 50-500x |

---

## Architecture Principles

### 1. Vector-Only Modeling

All data is represented as embeddings for unified similarity search:

```rust
// Core principle: Everything is a vector
trait Embeddable {
    fn dimension(&self) -> usize;
    fn embed(&self) -> Vec<f32>;
    fn from_embedding(vec: &[f32]) -> Result<Self, EmbedError>;
}
```

**Benefits:**
- Unified query interface
- Semantic similarity search
- Cross-domain pattern matching
- Hardware-friendly (SIMD, GPU)

### 2. Three-Tier Memory Hierarchy

```
┌─────────────────────────────────────────────┐
│  L1: Hot Cache (In-Memory, <1μs)           │
│  - Current positions, active orders         │
│  - Real-time market state                   │
│  - TTL-based eviction                       │
├─────────────────────────────────────────────┤
│  L2: AgentDB VectorDB (HNSW, <1ms)         │
│  - Historical observations                  │
│  - Strategy patterns                        │
│  - Reflexion traces                         │
├─────────────────────────────────────────────┤
│  L3: Cold Storage (Compressed, >10ms)      │
│  - Long-term backtests                      │
│  - Audit logs                               │
│  - Model checkpoints                        │
└─────────────────────────────────────────────┘
```

### 3. Deterministic Embeddings

Use hash-based embeddings for reproducibility:

```rust
use seahash::SeaHasher;

fn hash_embed(data: &[u8], dimension: usize) -> Vec<f32> {
    let mut hasher = SeaHasher::new();
    let mut embedding = vec![0.0; dimension];

    for i in 0..dimension {
        hasher.write_u64(i as u64);
        hasher.write(data);
        let hash = hasher.finish();
        embedding[i] = (hash as f32 / u64::MAX as f32) * 2.0 - 1.0;
        hasher = SeaHasher::new();
    }

    embedding
}
```

**Properties:**
- Deterministic: Same input → same embedding
- Fast: No neural network inference
- Collision-resistant: Cryptographic hash function
- Reversible: Store original data separately

---

## Schema Design

### Core Data Structures

#### 1. Market Observation

```rust
use uuid::Uuid;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// Unique identifier
    pub id: Uuid,

    /// Timestamp in microseconds since epoch
    pub timestamp_us: i64,

    /// Trading symbol (e.g., "AAPL", "BTC-USD")
    pub symbol: String,

    /// Price in quote currency
    pub price: f64,

    /// Volume in base currency
    pub volume: f64,

    /// Bid-ask spread
    pub spread: f64,

    /// Order book depth (top 5 levels)
    pub book_depth: BookDepth,

    /// 512-dimensional embedding (deterministic hash)
    pub embedding: Vec<f32>,

    /// Additional metadata (JSON)
    pub metadata: serde_json::Value,

    /// Data provenance
    pub provenance: Provenance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookDepth {
    pub bids: Vec<(f64, f64)>, // (price, volume)
    pub asks: Vec<(f64, f64)>,
}

impl Embeddable for Observation {
    fn dimension(&self) -> usize { 512 }

    fn embed(&self) -> Vec<f32> {
        let mut data = Vec::new();
        data.extend_from_slice(&self.timestamp_us.to_le_bytes());
        data.extend_from_slice(self.symbol.as_bytes());
        data.extend_from_slice(&self.price.to_le_bytes());
        data.extend_from_slice(&self.volume.to_le_bytes());
        data.extend_from_slice(&self.spread.to_le_bytes());

        hash_embed(&data, 512)
    }

    fn from_embedding(_vec: &[f32]) -> Result<Self, EmbedError> {
        Err(EmbedError::NotReversible)
    }
}
```

**AgentDB Configuration:**
```rust
let obs_db = VectorDBConfig::new()
    .dimension(512)
    .index_type(IndexType::HNSW)
    .hnsw_params(HNSWParams {
        m: 16,              // Max connections per layer
        ef_construction: 200, // Search depth during build
        ef_search: 50,       // Search depth during query
    })
    .quantization(Quantization::Scalar)
    .build()?;
```

#### 2. Trading Signal

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub id: Uuid,

    /// Strategy that generated this signal
    pub strategy_id: String,

    /// When signal was generated (μs)
    pub timestamp_us: i64,

    /// Target symbol
    pub symbol: String,

    /// Trade direction
    pub direction: Direction,

    /// Confidence score [0.0, 1.0]
    pub confidence: f64,

    /// Feature vector used for decision
    pub features: Vec<f32>,

    /// Human-readable reasoning
    pub reasoning: String,

    /// Causal links to observations
    pub causal_links: Vec<Uuid>,

    /// 768-dimensional embedding
    pub embedding: Vec<f32>,

    /// Provenance chain
    pub provenance: Provenance,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Direction {
    Long,
    Short,
    Neutral,
    Close,
}

impl Embeddable for Signal {
    fn dimension(&self) -> usize { 768 }

    fn embed(&self) -> Vec<f32> {
        let mut data = Vec::new();
        data.extend_from_slice(&self.timestamp_us.to_le_bytes());
        data.extend_from_slice(self.strategy_id.as_bytes());
        data.extend_from_slice(self.symbol.as_bytes());
        data.push(self.direction as u8);
        data.extend_from_slice(&self.confidence.to_le_bytes());

        // Include feature vector in embedding
        for &f in &self.features {
            data.extend_from_slice(&f.to_le_bytes());
        }

        hash_embed(&data, 768)
    }

    fn from_embedding(_vec: &[f32]) -> Result<Self, EmbedError> {
        Err(EmbedError::NotReversible)
    }
}
```

**AgentDB Configuration:**
```rust
let signal_db = VectorDBConfig::new()
    .dimension(768)
    .index_type(IndexType::HNSW)
    .hnsw_params(HNSWParams {
        m: 32,              // Higher connectivity for complex patterns
        ef_construction: 400,
        ef_search: 100,
    })
    .quantization(Quantization::Scalar)
    .metadata_index(true)  // Enable filtering by strategy_id
    .build()?;
```

#### 3. Order Lifecycle

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: Uuid,

    /// Source signal
    pub signal_id: Uuid,

    pub symbol: String,
    pub side: Side,
    pub quantity: f64,
    pub order_type: OrderType,
    pub limit_price: Option<f64>,
    pub stop_price: Option<f64>,

    /// Current status
    pub status: OrderStatus,

    /// Lifecycle timestamps
    pub timestamps: OrderTimestamps,

    /// Fill information
    pub fills: Vec<Fill>,

    /// 256-dimensional embedding
    pub embedding: Vec<f32>,

    /// Provenance
    pub provenance: Provenance,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Side { Buy, Sell }

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OrderType { Market, Limit, StopLoss, StopLimit }

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OrderStatus {
    Pending,
    Submitted,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderTimestamps {
    pub created_us: i64,
    pub submitted_us: Option<i64>,
    pub first_fill_us: Option<i64>,
    pub completed_us: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    pub timestamp_us: i64,
    pub quantity: f64,
    pub price: f64,
    pub fee: f64,
}

impl Embeddable for Order {
    fn dimension(&self) -> usize { 256 }

    fn embed(&self) -> Vec<f32> {
        let mut data = Vec::new();
        data.extend_from_slice(&self.signal_id.as_bytes());
        data.extend_from_slice(self.symbol.as_bytes());
        data.push(self.side as u8);
        data.push(self.order_type as u8);
        data.extend_from_slice(&self.quantity.to_le_bytes());

        if let Some(price) = self.limit_price {
            data.extend_from_slice(&price.to_le_bytes());
        }

        hash_embed(&data, 256)
    }

    fn from_embedding(_vec: &[f32]) -> Result<Self, EmbedError> {
        Err(EmbedError::NotReversible)
    }
}
```

**AgentDB Configuration:**
```rust
let order_db = VectorDBConfig::new()
    .dimension(256)
    .index_type(IndexType::HNSW)
    .hnsw_params(HNSWParams {
        m: 8,               // Lower connectivity for simple patterns
        ef_construction: 100,
        ef_search: 30,
    })
    .quantization(Quantization::Binary)  // 32x memory reduction
    .metadata_index(true)
    .build()?;
```

#### 4. ReasoningBank Reflexion Trace

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflexionTrace {
    pub id: Uuid,

    /// Decision being reflected upon
    pub decision_id: Uuid,

    /// Decision type (signal, order, strategy)
    pub decision_type: DecisionType,

    /// Trajectory: sequence of state-action pairs
    pub trajectory: Vec<StateAction>,

    /// Verdict: performance evaluation
    pub verdict: Verdict,

    /// Learned patterns extracted from trajectory
    pub learned_patterns: Vec<Pattern>,

    /// Counterfactual analysis
    pub counterfactuals: Vec<Counterfactual>,

    /// 1024-dimensional embedding
    pub embedding: Vec<f32>,

    /// Provenance
    pub provenance: Provenance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionType {
    Signal,
    Order,
    StrategySwitch,
    RiskLimit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateAction {
    pub timestamp_us: i64,
    pub state: State,
    pub action: Action,
    pub reward: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    pub positions: Vec<Position>,
    pub cash: f64,
    pub unrealized_pnl: f64,
    pub market_features: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub avg_price: f64,
    pub unrealized_pnl: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub action_type: ActionType,
    pub symbol: String,
    pub quantity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionType {
    Buy,
    Sell,
    Hold,
    ClosePosition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Verdict {
    /// Overall success score [-1.0, 1.0]
    pub score: f64,

    /// Return on investment
    pub roi: f64,

    /// Sharpe ratio
    pub sharpe: f64,

    /// Max drawdown
    pub max_drawdown: f64,

    /// Human-readable explanation
    pub explanation: String,

    /// What went well
    pub successes: Vec<String>,

    /// What went wrong
    pub failures: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub pattern_type: PatternType,
    pub description: String,
    pub confidence: f64,
    pub occurrences: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    EntryTiming,
    ExitTiming,
    RiskManagement,
    MarketRegime,
    Correlation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Counterfactual {
    pub description: String,
    pub alternative_action: Action,
    pub estimated_outcome: f64,
    pub probability: f64,
}

impl Embeddable for ReflexionTrace {
    fn dimension(&self) -> usize { 1024 }

    fn embed(&self) -> Vec<f32> {
        let mut data = Vec::new();
        data.extend_from_slice(&self.decision_id.as_bytes());
        data.push(self.decision_type as u8);
        data.extend_from_slice(&self.verdict.score.to_le_bytes());
        data.extend_from_slice(&self.verdict.roi.to_le_bytes());
        data.extend_from_slice(&self.verdict.sharpe.to_le_bytes());

        // Include trajectory summary
        for sa in &self.trajectory {
            data.extend_from_slice(&sa.reward.to_le_bytes());
        }

        hash_embed(&data, 1024)
    }

    fn from_embedding(_vec: &[f32]) -> Result<Self, EmbedError> {
        Err(EmbedError::NotReversible)
    }
}
```

**AgentDB Configuration:**
```rust
let reflexion_db = VectorDBConfig::new()
    .dimension(1024)
    .index_type(IndexType::HNSW)
    .hnsw_params(HNSWParams {
        m: 64,              // Highest connectivity for complex reasoning
        ef_construction: 800,
        ef_search: 200,
    })
    .quantization(Quantization::Scalar)
    .metadata_index(true)
    .build()?;
```

---

## Memory Patterns

### 1. Session Memory (L1 Cache)

Hot in-memory cache for real-time trading state.

```rust
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

pub struct SessionMemory {
    /// Current positions by symbol
    positions: Arc<RwLock<HashMap<String, Position>>>,

    /// Active orders by ID
    orders: Arc<RwLock<HashMap<Uuid, Order>>>,

    /// Recent observations (ring buffer)
    observations: Arc<RwLock<RingBuffer<Observation>>>,

    /// Active signals
    signals: Arc<RwLock<HashMap<Uuid, Signal>>>,

    /// Market state cache with TTL
    market_cache: Arc<RwLock<TtlCache<String, MarketState>>>,
}

impl SessionMemory {
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            positions: Arc::new(RwLock::new(HashMap::new())),
            orders: Arc::new(RwLock::new(HashMap::new())),
            observations: Arc::new(RwLock::new(RingBuffer::new(capacity))),
            signals: Arc::new(RwLock::new(HashMap::new())),
            market_cache: Arc::new(RwLock::new(TtlCache::new(ttl))),
        }
    }

    /// Get current position for symbol (zero-copy)
    pub fn get_position(&self, symbol: &str) -> Option<Position> {
        self.positions.read().unwrap().get(symbol).cloned()
    }

    /// Update position (< 100ns)
    pub fn update_position(&self, symbol: String, position: Position) {
        self.positions.write().unwrap().insert(symbol, position);
    }

    /// Get recent observations for symbol (< 1μs)
    pub fn get_recent_observations(&self, symbol: &str, n: usize) -> Vec<Observation> {
        self.observations
            .read()
            .unwrap()
            .iter()
            .filter(|obs| obs.symbol == symbol)
            .take(n)
            .cloned()
            .collect()
    }

    /// Check if order is active (< 50ns)
    pub fn is_order_active(&self, order_id: &Uuid) -> bool {
        self.orders
            .read()
            .unwrap()
            .get(order_id)
            .map(|o| matches!(o.status, OrderStatus::Pending | OrderStatus::Submitted))
            .unwrap_or(false)
    }
}

/// TTL cache for market state
struct TtlCache<K, V> {
    cache: HashMap<K, (V, Instant)>,
    ttl: Duration,
}

impl<K: Eq + std::hash::Hash, V> TtlCache<K, V> {
    fn new(ttl: Duration) -> Self {
        Self {
            cache: HashMap::new(),
            ttl,
        }
    }

    fn get(&mut self, key: &K) -> Option<&V> {
        if let Some((value, inserted)) = self.cache.get(key) {
            if inserted.elapsed() < self.ttl {
                return Some(value);
            }
            // Expired, remove lazily
            self.cache.remove(key);
        }
        None
    }

    fn insert(&mut self, key: K, value: V) {
        self.cache.insert(key, (value, Instant::now()));
    }
}

/// Ring buffer for efficient recent data
struct RingBuffer<T> {
    buffer: Vec<T>,
    head: usize,
    capacity: usize,
}

impl<T: Clone> RingBuffer<T> {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            head: 0,
            capacity,
        }
    }

    fn push(&mut self, item: T) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(item);
        } else {
            self.buffer[self.head] = item;
            self.head = (self.head + 1) % self.capacity;
        }
    }

    fn iter(&self) -> impl Iterator<Item = &T> {
        let (first, second) = self.buffer.split_at(self.head);
        second.iter().chain(first.iter())
    }
}
```

**Performance Targets:**
- Position lookup: <100ns
- Order status check: <50ns
- Recent observations: <1μs
- Cache hit rate: >95%

### 2. Long-Term Storage (L2 - AgentDB)

Historical data with vector similarity search.

```rust
use agentdb::{VectorDB, Query, Filter};

pub struct LongTermMemory {
    observations_db: VectorDB,
    signals_db: VectorDB,
    orders_db: VectorDB,
    reflexion_db: VectorDB,
}

impl LongTermMemory {
    pub async fn new() -> Result<Self, AgentDBError> {
        Ok(Self {
            observations_db: Self::init_observations_db().await?,
            signals_db: Self::init_signals_db().await?,
            orders_db: Self::init_orders_db().await?,
            reflexion_db: Self::init_reflexion_db().await?,
        })
    }

    /// Store observation (async, <5ms)
    pub async fn store_observation(&self, obs: &Observation) -> Result<(), AgentDBError> {
        self.observations_db
            .insert(obs.id.as_bytes(), &obs.embedding, Some(obs))
            .await
    }

    /// Find similar market conditions (< 1ms target)
    pub async fn find_similar_conditions(
        &self,
        query_obs: &Observation,
        k: usize,
        time_range: Option<(i64, i64)>,
    ) -> Result<Vec<Observation>, AgentDBError> {
        let mut query = Query::new(&query_obs.embedding).k(k);

        if let Some((start, end)) = time_range {
            query = query.filter(Filter::and(vec![
                Filter::gte("timestamp_us", start),
                Filter::lte("timestamp_us", end),
            ]));
        }

        self.observations_db.search(query).await
    }

    /// Get signals by strategy (< 1ms)
    pub async fn get_signals_by_strategy(
        &self,
        strategy_id: &str,
        limit: usize,
    ) -> Result<Vec<Signal>, AgentDBError> {
        let query = Query::new_filter(
            Filter::eq("strategy_id", strategy_id)
        ).limit(limit);

        self.signals_db.search(query).await
    }

    /// Find similar trading decisions (< 1ms)
    pub async fn find_similar_decisions(
        &self,
        signal: &Signal,
        k: usize,
    ) -> Result<Vec<Signal>, AgentDBError> {
        self.signals_db
            .search(Query::new(&signal.embedding).k(k))
            .await
    }

    /// Store reflexion trace (< 5ms)
    pub async fn store_reflexion(&self, trace: &ReflexionTrace) -> Result<(), AgentDBError> {
        self.reflexion_db
            .insert(trace.id.as_bytes(), &trace.embedding, Some(trace))
            .await
    }

    /// Find similar past decisions to learn from (< 1ms)
    pub async fn find_similar_traces(
        &self,
        trace: &ReflexionTrace,
        k: usize,
        min_score: f64,
    ) -> Result<Vec<ReflexionTrace>, AgentDBError> {
        let query = Query::new(&trace.embedding)
            .k(k)
            .filter(Filter::gte("verdict.score", min_score));

        self.reflexion_db.search(query).await
    }
}
```

**Performance Targets:**
- Vector search: <1ms
- Filtered search: <2ms
- Batch insert: <10ms (1000 items)
- Storage overhead: 4-32x reduction via quantization

### 3. Reflexion Memory (ReasoningBank Pattern)

Continuous learning through reflection loops.

```rust
use async_trait::async_trait;

#[async_trait]
pub trait ReflexionEngine {
    async fn reflect(&self, decision_id: Uuid) -> Result<ReflexionTrace, ReflexionError>;
    async fn learn_patterns(&self, trace: &ReflexionTrace) -> Result<Vec<Pattern>, ReflexionError>;
    async fn apply_patterns(&self, signal: &mut Signal) -> Result<(), ReflexionError>;
}

pub struct AgentDBReflexionEngine {
    memory: Arc<LongTermMemory>,
    session: Arc<SessionMemory>,
}

impl AgentDBReflexionEngine {
    /// Reflect on past decision
    pub async fn reflect_on_signal(&self, signal_id: Uuid) -> Result<ReflexionTrace, ReflexionError> {
        // 1. Retrieve signal and related orders
        let signal = self.session.get_signal(&signal_id)
            .ok_or(ReflexionError::SignalNotFound)?;

        let orders = self.session.get_orders_for_signal(&signal_id);

        // 2. Build trajectory
        let trajectory = self.build_trajectory(&signal, &orders).await?;

        // 3. Calculate verdict
        let verdict = self.calculate_verdict(&trajectory)?;

        // 4. Extract patterns
        let learned_patterns = self.extract_patterns(&trajectory, &verdict).await?;

        // 5. Generate counterfactuals
        let counterfactuals = self.generate_counterfactuals(&trajectory).await?;

        // 6. Create and store trace
        let trace = ReflexionTrace {
            id: Uuid::new_v4(),
            decision_id: signal_id,
            decision_type: DecisionType::Signal,
            trajectory,
            verdict,
            learned_patterns,
            counterfactuals,
            embedding: Vec::new(), // Will be computed
            provenance: Provenance::new("reflexion_engine"),
        };

        let mut trace = trace;
        trace.embedding = trace.embed();

        self.memory.store_reflexion(&trace).await?;

        Ok(trace)
    }

    /// Learn from similar past experiences
    pub async fn learn_from_history(&self, current_signal: &Signal) -> Result<Vec<Pattern>, ReflexionError> {
        // Find similar past signals
        let similar_signals = self.memory
            .find_similar_decisions(current_signal, 10)
            .await?;

        // Get reflexion traces for those signals
        let mut all_patterns = Vec::new();

        for signal in similar_signals {
            if let Ok(traces) = self.memory
                .find_similar_traces_by_decision(signal.id, 0.5)
                .await
            {
                for trace in traces {
                    all_patterns.extend(trace.learned_patterns.clone());
                }
            }
        }

        // Deduplicate and rank patterns
        let patterns = self.deduplicate_patterns(all_patterns);

        Ok(patterns)
    }

    /// Apply learned patterns to new signal
    pub async fn enhance_signal(&self, signal: &mut Signal) -> Result<(), ReflexionError> {
        // Get relevant patterns
        let patterns = self.learn_from_history(signal).await?;

        // Apply pattern adjustments
        for pattern in patterns {
            match pattern.pattern_type {
                PatternType::EntryTiming => {
                    self.adjust_entry_timing(signal, &pattern)?;
                }
                PatternType::RiskManagement => {
                    self.adjust_risk_parameters(signal, &pattern)?;
                }
                _ => {}
            }
        }

        // Update signal reasoning
        signal.reasoning.push_str(&format!(
            "\n\n[ReflexionBank] Applied {} learned patterns",
            patterns.len()
        ));

        Ok(())
    }
}
```

**Learning Loop:**

```
┌─────────────────────────────────────────────┐
│  1. Generate Signal                         │
└────────────┬────────────────────────────────┘
             │
             v
┌─────────────────────────────────────────────┐
│  2. Enhance with Learned Patterns           │
│     (AgentDB similarity search)             │
└────────────┬────────────────────────────────┘
             │
             v
┌─────────────────────────────────────────────┐
│  3. Execute Trade                           │
└────────────┬────────────────────────────────┘
             │
             v
┌─────────────────────────────────────────────┐
│  4. Observe Outcome                         │
└────────────┬────────────────────────────────┘
             │
             v
┌─────────────────────────────────────────────┐
│  5. Reflect & Generate Trace                │
│     - Build trajectory                      │
│     - Calculate verdict                     │
│     - Extract patterns                      │
│     - Generate counterfactuals              │
└────────────┬────────────────────────────────┘
             │
             v
┌─────────────────────────────────────────────┐
│  6. Store in AgentDB                        │
│     (Available for future decisions)        │
└─────────────────────────────────────────────┘
```

---

## AgentDB Integration

### Client Configuration

```rust
// Cargo.toml
[dependencies]
agentdb-client = "0.3"
tokio = { version = "1.35", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
uuid = { version = "1.6", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }
anyhow = "1.0"
thiserror = "1.0"
```

### Client Setup

```rust
use agentdb_client::{AgentDBClient, Config};
use std::time::Duration;

pub async fn init_agentdb() -> Result<AgentDBClient, anyhow::Error> {
    let config = Config {
        endpoint: std::env::var("AGENTDB_ENDPOINT")
            .unwrap_or_else(|_| "http://localhost:8080".to_string()),

        // Connection pooling
        max_connections: 100,
        connection_timeout: Duration::from_secs(5),
        request_timeout: Duration::from_secs(30),

        // Retry logic
        max_retries: 3,
        retry_delay: Duration::from_millis(100),

        // Batching
        batch_size: 1000,
        batch_timeout: Duration::from_millis(10),

        // TLS
        tls_enabled: true,
        tls_cert_path: Some("certs/agentdb.crt".into()),
    };

    AgentDBClient::new(config).await
}
```

### Batch Operations

```rust
/// Batch insert observations (10x faster than individual inserts)
pub async fn batch_insert_observations(
    db: &VectorDB,
    observations: Vec<Observation>,
) -> Result<(), AgentDBError> {
    let batch: Vec<_> = observations
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
    Ok(())
}

/// Batch query for multiple symbols
pub async fn batch_query_symbols(
    db: &VectorDB,
    symbols: Vec<String>,
    query_vec: &[f32],
    k: usize,
) -> Result<HashMap<String, Vec<Observation>>, AgentDBError> {
    let queries: Vec<_> = symbols
        .iter()
        .map(|symbol| {
            Query::new(query_vec)
                .k(k)
                .filter(Filter::eq("symbol", symbol))
        })
        .collect();

    let results = db.batch_search(queries).await?;

    let mut map = HashMap::new();
    for (symbol, obs_vec) in symbols.into_iter().zip(results) {
        map.insert(symbol, obs_vec);
    }

    Ok(map)
}
```

### Zero-Copy Deserialization

```rust
use bytes::Bytes;
use serde::de::DeserializeOwned;

/// Zero-copy reader for AgentDB results
pub struct ZeroCopyReader {
    buffer: Bytes,
}

impl ZeroCopyReader {
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            buffer: Bytes::from(data),
        }
    }

    /// Deserialize without copying
    pub fn deserialize<T: DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_slice(&self.buffer)
    }

    /// View raw bytes
    pub fn as_slice(&self) -> &[u8] {
        &self.buffer
    }
}
```

---

## Query Patterns

### 1. Vector Similarity Search

**Use Case:** Find similar market conditions

```rust
/// Query: Given current market state, find similar historical states
pub async fn find_similar_market_states(
    db: &VectorDB,
    current: &Observation,
    k: usize,
    time_window_hours: Option<i64>,
) -> Result<Vec<Observation>, AgentDBError> {
    let mut query = Query::new(&current.embedding)
        .k(k)
        .filter(Filter::eq("symbol", &current.symbol));

    // Optional: Restrict to recent history
    if let Some(hours) = time_window_hours {
        let cutoff = current.timestamp_us - (hours * 3600 * 1_000_000);
        query = query.filter(Filter::gte("timestamp_us", cutoff));
    }

    db.search(query).await
}
```

**Performance Target:** <1ms for k=10, <2ms for k=100

### 2. Temporal Range Query

**Use Case:** Get all observations in time window

```rust
/// Query: Get observations for symbol in time range
pub async fn get_observations_in_range(
    db: &VectorDB,
    symbol: &str,
    start_us: i64,
    end_us: i64,
) -> Result<Vec<Observation>, AgentDBError> {
    let query = Query::new_filter(
        Filter::and(vec![
            Filter::eq("symbol", symbol),
            Filter::gte("timestamp_us", start_us),
            Filter::lte("timestamp_us", end_us),
        ])
    ).limit(10000);

    db.search(query).await
}
```

**Performance Target:** <5ms for 1-hour window, <50ms for 1-day window

### 3. Strategy Performance Lookup

**Use Case:** Retrieve signals by strategy

```rust
/// Query: Get top-performing signals for strategy
pub async fn get_top_signals_for_strategy(
    db: &VectorDB,
    strategy_id: &str,
    min_confidence: f64,
    limit: usize,
) -> Result<Vec<Signal>, AgentDBError> {
    let query = Query::new_filter(
        Filter::and(vec![
            Filter::eq("strategy_id", strategy_id),
            Filter::gte("confidence", min_confidence),
        ])
    )
    .limit(limit)
    .sort_by("confidence", SortOrder::Desc);

    db.search(query).await
}
```

**Performance Target:** <1ms

### 4. Provenance Chain Traversal

**Use Case:** Trace decision ancestry

```rust
/// Query: Get full provenance chain for decision
pub async fn get_provenance_chain(
    memory: &LongTermMemory,
    decision_id: Uuid,
) -> Result<ProvenanceChain, AgentDBError> {
    let mut chain = ProvenanceChain::new();
    let mut current_id = decision_id;

    // Traverse backwards through causal links
    for _ in 0..100 {  // Max depth
        if let Some(signal) = memory.get_signal(current_id).await? {
            chain.add(signal.clone());

            // Get causal observations
            for obs_id in &signal.causal_links {
                if let Some(obs) = memory.get_observation(*obs_id).await? {
                    chain.add(obs.clone());
                }
            }

            // Check for parent decision
            if let Some(parent_id) = signal.provenance.parent_id {
                current_id = parent_id;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    Ok(chain)
}

pub struct ProvenanceChain {
    signals: Vec<Signal>,
    observations: Vec<Observation>,
    depth: usize,
}
```

**Performance Target:** <10ms for depth=10

### 5. Performance Aggregation

**Use Case:** Calculate strategy metrics

```rust
/// Query: Aggregate performance metrics for strategy
pub async fn calculate_strategy_metrics(
    memory: &LongTermMemory,
    strategy_id: &str,
    time_range: (i64, i64),
) -> Result<StrategyMetrics, AgentDBError> {
    // Get all reflexion traces for strategy
    let traces = memory
        .get_reflexions_by_strategy(strategy_id, time_range)
        .await?;

    let mut metrics = StrategyMetrics::default();

    for trace in traces {
        metrics.total_trades += 1;
        metrics.total_pnl += trace.verdict.roi;
        metrics.sharpe_sum += trace.verdict.sharpe;

        if trace.verdict.score > 0.0 {
            metrics.winning_trades += 1;
        }

        if trace.verdict.max_drawdown < metrics.max_drawdown {
            metrics.max_drawdown = trace.verdict.max_drawdown;
        }
    }

    metrics.win_rate = metrics.winning_trades as f64 / metrics.total_trades as f64;
    metrics.avg_sharpe = metrics.sharpe_sum / metrics.total_trades as f64;

    Ok(metrics)
}

#[derive(Debug, Default)]
pub struct StrategyMetrics {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub win_rate: f64,
    pub total_pnl: f64,
    pub avg_sharpe: f64,
    pub sharpe_sum: f64,
    pub max_drawdown: f64,
}
```

**Performance Target:** <50ms for 1000 trades

---

## Provenance & Cryptography

### Provenance Tracking

```rust
use ed25519_dalek::{Keypair, PublicKey, Signature, Signer, Verifier};
use sha2::{Sha256, Digest};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    /// Creator identity
    pub creator: String,

    /// Creation timestamp (μs)
    pub created_us: i64,

    /// Parent decision ID (for causal chains)
    pub parent_id: Option<Uuid>,

    /// Cryptographic signature
    pub signature: Vec<u8>,

    /// Public key of signer
    pub public_key: Vec<u8>,

    /// Hash of signed data
    pub hash: Vec<u8>,
}

impl Provenance {
    /// Create new provenance with signature
    pub fn new(creator: &str) -> Self {
        let created_us = chrono::Utc::now().timestamp_micros();

        Self {
            creator: creator.to_string(),
            created_us,
            parent_id: None,
            signature: Vec::new(),
            public_key: Vec::new(),
            hash: Vec::new(),
        }
    }

    /// Sign data with keypair
    pub fn sign<T: Serialize>(mut self, data: &T, keypair: &Keypair) -> Result<Self, ProvenanceError> {
        // Serialize data
        let data_bytes = serde_json::to_vec(data)?;

        // Hash data
        let mut hasher = Sha256::new();
        hasher.update(&data_bytes);
        hasher.update(&self.creator.as_bytes());
        hasher.update(&self.created_us.to_le_bytes());
        self.hash = hasher.finalize().to_vec();

        // Sign hash
        let signature = keypair.sign(&self.hash);
        self.signature = signature.to_bytes().to_vec();
        self.public_key = keypair.public.to_bytes().to_vec();

        Ok(self)
    }

    /// Verify signature
    pub fn verify(&self) -> Result<bool, ProvenanceError> {
        let public_key = PublicKey::from_bytes(&self.public_key)?;
        let signature = Signature::from_bytes(&self.signature)?;

        Ok(public_key.verify(&self.hash, &signature).is_ok())
    }

    /// Link to parent decision
    pub fn with_parent(mut self, parent_id: Uuid) -> Self {
        self.parent_id = Some(parent_id);
        self
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ProvenanceError {
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Invalid public key")]
    InvalidPublicKey,

    #[error("Invalid signature")]
    InvalidSignature,

    #[error("Verification failed")]
    VerificationFailed,
}
```

### Usage Example

```rust
/// Sign observation with keypair
pub fn create_signed_observation(
    obs: Observation,
    keypair: &Keypair,
) -> Result<Observation, ProvenanceError> {
    let provenance = Provenance::new("market_data_collector")
        .sign(&obs, keypair)?;

    let mut signed_obs = obs;
    signed_obs.provenance = provenance;

    Ok(signed_obs)
}

/// Verify observation authenticity
pub fn verify_observation(obs: &Observation) -> Result<bool, ProvenanceError> {
    obs.provenance.verify()
}
```

---

## Performance Targets

### Latency Budgets

| Operation | Target | Allowable | Critical Path |
|-----------|--------|-----------|---------------|
| L1 position lookup | <100ns | 500ns | Yes |
| L1 order check | <50ns | 200ns | Yes |
| L2 vector search (k=10) | <1ms | 5ms | Yes |
| L2 filtered search | <2ms | 10ms | No |
| L2 batch insert (1000) | <10ms | 50ms | No |
| L3 cold storage read | <100ms | 1s | No |
| Reflexion trace generation | <50ms | 200ms | No |
| Pattern learning | <100ms | 500ms | No |

### Throughput Targets

| Metric | Target |
|--------|--------|
| Observations/sec | >100,000 |
| Signals/sec | >10,000 |
| Orders/sec | >5,000 |
| Vector searches/sec | >10,000 |
| Reflexion traces/hour | >1,000 |

### Memory Targets

| Component | Baseline | With Quantization | Reduction |
|-----------|----------|-------------------|-----------|
| Observations (1M) | 2.0 GB | 500 MB | 4x |
| Signals (100K) | 768 MB | 192 MB | 4x |
| Orders (100K) | 256 MB | 8 MB | 32x (binary) |
| Reflexion (10K) | 1.0 GB | 250 MB | 4x |
| **Total** | **4.0 GB** | **950 MB** | **4.2x** |

---

## Rust Implementation

### Project Structure

```
neural-trader-rust/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── memory/
│   │   ├── mod.rs
│   │   ├── session.rs       # L1 cache
│   │   ├── longterm.rs      # L2 AgentDB
│   │   ├── reflexion.rs     # ReasoningBank
│   │   └── provenance.rs    # Cryptographic tracking
│   ├── schema/
│   │   ├── mod.rs
│   │   ├── observation.rs
│   │   ├── signal.rs
│   │   ├── order.rs
│   │   └── trace.rs
│   ├── agentdb/
│   │   ├── mod.rs
│   │   ├── client.rs        # Client wrapper
│   │   ├── config.rs        # Configuration
│   │   └── query.rs         # Query builders
│   ├── embedding/
│   │   ├── mod.rs
│   │   ├── hash.rs          # Hash-based embeddings
│   │   └── traits.rs        # Embeddable trait
│   └── benchmark/
│       ├── mod.rs
│       ├── latency.rs
│       └── throughput.rs
└── tests/
    ├── integration_test.rs
    └── benchmark_test.rs
```

### Cargo.toml

```toml
[package]
name = "neural-trader-rust"
version = "0.1.0"
edition = "2021"

[dependencies]
# Async runtime
tokio = { version = "1.35", features = ["full"] }
async-trait = "0.1"

# AgentDB client
agentdb-client = "0.3"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
bincode = "1.3"

# Data structures
uuid = { version = "1.6", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }
dashmap = "5.5"  # Concurrent HashMap

# Cryptography
ed25519-dalek = "2.0"
sha2 = "0.10"
seahash = "4.1"

# Hashing
blake3 = "1.5"

# Error handling
anyhow = "1.0"
thiserror = "1.0"

# Metrics
prometheus = "0.13"

# Tracing
tracing = "0.1"
tracing-subscriber = "0.3"

[dev-dependencies]
criterion = "0.5"
proptest = "1.4"
tokio-test = "0.4"

[[bench]]
name = "memory_benchmark"
harness = false
```

### Main Entry Point

```rust
// src/main.rs
use neural_trader_rust::{
    memory::{SessionMemory, LongTermMemory, ReflexionEngine},
    agentdb::init_agentdb,
};
use std::time::Duration;
use tracing::{info, error};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    info!("Initializing Neural Trading System");

    // Initialize memory layers
    let session = SessionMemory::new(10000, Duration::from_secs(300));
    let longterm = LongTermMemory::new().await?;
    let reflexion = ReflexionEngine::new(session.clone(), longterm.clone());

    info!("Memory layers initialized");

    // Start trading loop
    loop {
        // Your trading logic here
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
```

### Benchmarking Harness

```rust
// benches/memory_benchmark.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use neural_trader_rust::{
    memory::SessionMemory,
    schema::Observation,
};
use std::time::Duration;

fn bench_session_memory(c: &mut Criterion) {
    let session = SessionMemory::new(10000, Duration::from_secs(300));

    let mut group = c.benchmark_group("session_memory");

    // Benchmark position lookup
    group.bench_function("position_lookup", |b| {
        b.iter(|| {
            black_box(session.get_position("AAPL"))
        });
    });

    // Benchmark order check
    group.bench_function("order_check", |b| {
        let order_id = uuid::Uuid::new_v4();
        b.iter(|| {
            black_box(session.is_order_active(&order_id))
        });
    });

    // Benchmark recent observations
    group.bench_function("recent_observations", |b| {
        b.iter(|| {
            black_box(session.get_recent_observations("AAPL", 100))
        });
    });

    group.finish();
}

criterion_group!(benches, bench_session_memory);
criterion_main!(benches);
```

---

## Benchmarking Strategy

### 1. Latency Benchmarks

```rust
use std::time::Instant;

pub struct LatencyBenchmark {
    measurements: Vec<Duration>,
}

impl LatencyBenchmark {
    pub fn new() -> Self {
        Self {
            measurements: Vec::with_capacity(10000),
        }
    }

    pub fn measure<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed();
        self.measurements.push(elapsed);
        result
    }

    pub fn percentile(&self, p: f64) -> Duration {
        let mut sorted = self.measurements.clone();
        sorted.sort();
        let idx = ((sorted.len() as f64) * p / 100.0) as usize;
        sorted[idx]
    }

    pub fn report(&self) {
        println!("Latency Benchmark Results:");
        println!("  p50: {:?}", self.percentile(50.0));
        println!("  p95: {:?}", self.percentile(95.0));
        println!("  p99: {:?}", self.percentile(99.0));
        println!("  p99.9: {:?}", self.percentile(99.9));
        println!("  max: {:?}", self.measurements.iter().max().unwrap());
    }
}
```

### 2. Throughput Benchmarks

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub struct ThroughputBenchmark {
    counter: Arc<AtomicUsize>,
    duration: Duration,
}

impl ThroughputBenchmark {
    pub fn new(duration: Duration) -> Self {
        Self {
            counter: Arc::new(AtomicUsize::new(0)),
            duration,
        }
    }

    pub async fn run<F, Fut>(&self, mut f: F) -> f64
    where
        F: FnMut() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send,
    {
        let counter = self.counter.clone();
        let duration = self.duration;

        let handle = tokio::spawn(async move {
            let start = Instant::now();
            while start.elapsed() < duration {
                f().await;
                counter.fetch_add(1, Ordering::Relaxed);
            }
        });

        handle.await.unwrap();

        let total = self.counter.load(Ordering::Relaxed);
        total as f64 / self.duration.as_secs_f64()
    }

    pub fn report(&self, ops_per_sec: f64) {
        println!("Throughput: {:.2} ops/sec", ops_per_sec);
    }
}
```

### 3. Memory Benchmark

```rust
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct MemoryTracker;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for MemoryTracker {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ret = System.alloc(layout);
        if !ret.is_null() {
            ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ret
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        ALLOCATED.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static GLOBAL: MemoryTracker = MemoryTracker;

pub fn current_memory_usage() -> usize {
    ALLOCATED.load(Ordering::Relaxed)
}
```

### 4. Integration Test Suite

```rust
// tests/integration_test.rs
use neural_trader_rust::{
    memory::{SessionMemory, LongTermMemory},
    schema::{Observation, Signal},
};
use std::time::Duration;

#[tokio::test]
async fn test_end_to_end_workflow() {
    // Initialize
    let session = SessionMemory::new(1000, Duration::from_secs(60));
    let longterm = LongTermMemory::new().await.unwrap();

    // Create observation
    let obs = Observation::new("AAPL", 150.0, 1000.0);

    // Store in session
    session.store_observation(obs.clone());

    // Store in long-term
    longterm.store_observation(&obs).await.unwrap();

    // Query similar
    let similar = longterm
        .find_similar_conditions(&obs, 10, None)
        .await
        .unwrap();

    assert!(!similar.is_empty());
}

#[tokio::test]
async fn test_performance_targets() {
    let session = SessionMemory::new(10000, Duration::from_secs(300));

    // Test position lookup < 1μs
    let start = std::time::Instant::now();
    for _ in 0..10000 {
        session.get_position("AAPL");
    }
    let elapsed = start.elapsed();
    assert!(elapsed.as_micros() < 10000); // <1μs per op
}
```

---

## Performance Optimization Checklist

- [ ] Use `#[inline]` for hot path functions
- [ ] Enable LTO in release builds (`lto = true`)
- [ ] Use `Arc` instead of `Rc` for thread safety
- [ ] Pool allocations for frequently created objects
- [ ] Use `SmallVec` for small collections
- [ ] Enable SIMD for vector operations
- [ ] Use `parking_lot` instead of `std::sync` for faster locks
- [ ] Profile with `perf` and `flamegraph`
- [ ] Benchmark with `criterion`
- [ ] Test with `miri` for undefined behavior

---

## Next Steps

1. **Implement Core Schema** (Week 1)
   - Define all Rust structs
   - Implement `Embeddable` trait
   - Add serialization

2. **Build Memory Layers** (Week 2)
   - SessionMemory with TTL cache
   - LongTermMemory with AgentDB
   - ReflexionEngine

3. **Integration Testing** (Week 3)
   - End-to-end workflows
   - Performance validation
   - Stress testing

4. **Optimization** (Week 4)
   - Profile hot paths
   - Optimize allocations
   - Tune HNSW parameters

5. **Production Deployment** (Week 5)
   - Docker containerization
   - Monitoring setup
   - Rollout plan

---

## References

- [AgentDB Documentation](https://github.com/agentdb/agentdb)
- [HNSW Paper](https://arxiv.org/abs/1603.09320)
- [ReasoningBank](https://arxiv.org/abs/2305.17888)
- [Tokio Async Runtime](https://tokio.rs)
- [Criterion Benchmarking](https://bheisler.github.io/criterion.rs/book/)

---

**Status:** Design Complete ✅
**Next:** Implementation Phase
**Owner:** ML Model Developer
**Last Updated:** 2025-11-12
