# Neural Trading Rust Port - Memory Architecture and AgentDB Integration

**Version:** 1.0.0
**Date:** 2025-11-12
**Status:** Design Complete
**Cross-References:** [Architecture](03_Architecture.md) | [Parity](02_Parity_Requirements.md) | [AgentDB Docs](../../docs/RUST_AGENTDB_MEMORY_ARCHITECTURE.md)

---

## Table of Contents

1. [Schema Definitions](#schema-definitions)
2. [HNSW Index Configuration](#hnsw-index-configuration)
3. [Deterministic Hash-Based Embeddings](#deterministic-hash-based-embeddings)
4. [Provenance Tracking with Ed25519](#provenance-tracking-with-ed25519)
5. [Query Templates with Performance Targets](#query-templates-with-performance-targets)
6. [Reflexion Loop Integration](#reflexion-loop-integration)
7. [Memory Hierarchy](#memory-hierarchy)
8. [Rust Client Implementation](#rust-client-implementation)

---

## Schema Definitions

### 1. Market Observation Schema

**Purpose:** Store raw market data with deterministic embeddings for similarity search

```rust
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// Unique identifier
    pub id: Uuid,

    /// Timestamp in microseconds since epoch
    pub timestamp_us: i64,

    /// Trading symbol (e.g., "AAPL", "BTC-USD")
    pub symbol: String,

    /// Price in quote currency
    pub price: Decimal,

    /// Volume in base currency
    pub volume: Decimal,

    /// Bid-ask spread
    pub spread: Decimal,

    /// Order book depth (top 5 levels)
    pub book_depth: BookDepth,

    /// 512-dimensional embedding (deterministic hash)
    #[serde(skip)]
    pub embedding: Vec<f32>,

    /// Additional metadata (JSON)
    pub metadata: serde_json::Value,

    /// Data provenance
    pub provenance: Provenance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookDepth {
    /// Bid levels: [(price, volume), ...]
    pub bids: Vec<(Decimal, Decimal)>,

    /// Ask levels: [(price, volume), ...]
    pub asks: Vec<(Decimal, Decimal)>,
}

impl Observation {
    pub fn new(symbol: String, price: Decimal, volume: Decimal) -> Self {
        let id = Uuid::new_v4();
        let timestamp_us = Utc::now().timestamp_micros();

        let mut obs = Self {
            id,
            timestamp_us,
            symbol,
            price,
            volume,
            spread: Decimal::ZERO,
            book_depth: BookDepth {
                bids: Vec::new(),
                asks: Vec::new(),
            },
            embedding: Vec::new(),
            metadata: serde_json::json!({}),
            provenance: Provenance::new("market_data_collector"),
        };

        // Generate embedding
        obs.embedding = obs.compute_embedding();

        obs
    }

    /// Compute deterministic hash-based embedding
    pub fn compute_embedding(&self) -> Vec<f32> {
        let mut data = Vec::new();
        data.extend_from_slice(&self.timestamp_us.to_le_bytes());
        data.extend_from_slice(self.symbol.as_bytes());
        data.extend_from_slice(&self.price.to_string().as_bytes());
        data.extend_from_slice(&self.volume.to_string().as_bytes());
        data.extend_from_slice(&self.spread.to_string().as_bytes());

        hash_embed(&data, 512)
    }
}
```

**AgentDB Collection Configuration:**

```rust
use agentdb_client::{CollectionConfig, IndexType, HNSWParams, Quantization};

pub fn observation_collection_config() -> CollectionConfig {
    CollectionConfig::new("observations")
        .dimension(512)
        .index_type(IndexType::HNSW)
        .hnsw_params(HNSWParams {
            m: 16,                  // Max connections per layer
            ef_construction: 200,   // Search depth during build
            ef_search: 50,          // Search depth during query
        })
        .quantization(Quantization::Scalar)
        .metadata_index(vec!["symbol", "timestamp_us"])
        .build()
}
```

---

### 2. Trading Signal Schema

**Purpose:** Store generated trading signals with causal links to observations

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
    pub features: Vec<f64>,

    /// Human-readable reasoning
    pub reasoning: String,

    /// Causal links to observations
    pub causal_links: Vec<Uuid>,

    /// 768-dimensional embedding
    #[serde(skip)]
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

impl Signal {
    pub fn compute_embedding(&self) -> Vec<f32> {
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
}
```

**AgentDB Configuration:**

```rust
pub fn signal_collection_config() -> CollectionConfig {
    CollectionConfig::new("signals")
        .dimension(768)
        .index_type(IndexType::HNSW)
        .hnsw_params(HNSWParams {
            m: 32,                  // Higher connectivity for complex patterns
            ef_construction: 400,
            ef_search: 100,
        })
        .quantization(Quantization::Scalar)
        .metadata_index(vec!["strategy_id", "symbol", "confidence"])
        .build()
}
```

---

### 3. Order Lifecycle Schema

**Purpose:** Track order execution and fills

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: Uuid,

    /// Source signal
    pub signal_id: Uuid,

    pub symbol: String,
    pub side: OrderSide,
    pub quantity: u32,
    pub order_type: OrderType,
    pub limit_price: Option<Decimal>,
    pub stop_price: Option<Decimal>,

    /// Current status
    pub status: OrderStatus,

    /// Lifecycle timestamps
    pub timestamps: OrderTimestamps,

    /// Fill information
    pub fills: Vec<Fill>,

    /// 256-dimensional embedding
    #[serde(skip)]
    pub embedding: Vec<f32>,

    /// Provenance
    pub provenance: Provenance,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OrderSide { Buy, Sell }

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
    pub quantity: u32,
    pub price: Decimal,
    pub fee: Decimal,
}

impl Order {
    pub fn compute_embedding(&self) -> Vec<f32> {
        let mut data = Vec::new();
        data.extend_from_slice(self.signal_id.as_bytes());
        data.extend_from_slice(self.symbol.as_bytes());
        data.push(self.side as u8);
        data.push(self.order_type as u8);
        data.extend_from_slice(&self.quantity.to_le_bytes());

        if let Some(price) = self.limit_price {
            data.extend_from_slice(&price.to_string().as_bytes());
        }

        hash_embed(&data, 256)
    }
}
```

**AgentDB Configuration:**

```rust
pub fn order_collection_config() -> CollectionConfig {
    CollectionConfig::new("orders")
        .dimension(256)
        .index_type(IndexType::HNSW)
        .hnsw_params(HNSWParams {
            m: 8,               // Lower connectivity for simple patterns
            ef_construction: 100,
            ef_search: 30,
        })
        .quantization(Quantization::Binary)  // 32x memory reduction
        .metadata_index(vec!["symbol", "status", "signal_id"])
        .build()
}
```

---

### 4. ReasoningBank Reflexion Trace Schema

**Purpose:** Store decision trajectories and learned patterns for continuous improvement

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
    #[serde(skip)]
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
    pub state: PortfolioState,
    pub action: TradingAction,
    pub reward: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioState {
    pub positions: Vec<Position>,
    pub cash: Decimal,
    pub unrealized_pnl: Decimal,
    pub market_features: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingAction {
    pub action_type: ActionType,
    pub symbol: String,
    pub quantity: u32,
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
    pub alternative_action: TradingAction,
    pub estimated_outcome: f64,
    pub probability: f64,
}

impl ReflexionTrace {
    pub fn compute_embedding(&self) -> Vec<f32> {
        let mut data = Vec::new();
        data.extend_from_slice(self.decision_id.as_bytes());
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
}
```

**AgentDB Configuration:**

```rust
pub fn reflexion_collection_config() -> CollectionConfig {
    CollectionConfig::new("reflexion_traces")
        .dimension(1024)
        .index_type(IndexType::HNSW)
        .hnsw_params(HNSWParams {
            m: 64,                  // Highest connectivity for complex reasoning
            ef_construction: 800,
            ef_search: 200,
        })
        .quantization(Quantization::Scalar)
        .metadata_index(vec!["decision_type", "verdict.score"])
        .build()
}
```

---

## HNSW Index Configuration

### Performance Tuning Matrix

| Collection | Dimension | M | ef_construction | ef_search | Quantization | Target Latency |
|-----------|-----------|---|-----------------|-----------|--------------|----------------|
| Observations | 512 | 16 | 200 | 50 | Scalar | <1ms |
| Signals | 768 | 32 | 400 | 100 | Scalar | <1ms |
| Orders | 256 | 8 | 100 | 30 | Binary | <0.5ms |
| Reflexion | 1024 | 64 | 800 | 200 | Scalar | <2ms |

### Parameter Guidelines

**M (Max connections per layer):**
- Low (8-16): Simple patterns, fast inserts, lower memory
- Medium (32): Balanced complexity and performance
- High (64+): Complex patterns, slower inserts, higher recall

**ef_construction (Build-time search depth):**
- Low (100-200): Faster index building
- High (400-800): Better index quality, slower building

**ef_search (Query-time search depth):**
- Low (30-50): Faster queries, lower recall
- High (100-200): Slower queries, higher recall

**Quantization:**
- `Scalar`: 4x memory reduction, <1% accuracy loss
- `Binary`: 32x memory reduction, 5-10% accuracy loss
- `None`: No compression, highest accuracy

---

## Deterministic Hash-Based Embeddings

### Implementation

```rust
use seahash::SeaHasher;
use std::hash::{Hash, Hasher};

/// Generate deterministic hash-based embedding
pub fn hash_embed(data: &[u8], dimension: usize) -> Vec<f32> {
    let mut embedding = Vec::with_capacity(dimension);

    for i in 0..dimension {
        let mut hasher = SeaHasher::new();

        // Hash dimension index + data
        hasher.write_u64(i as u64);
        hasher.write(data);

        let hash = hasher.finish();

        // Convert to [-1.0, 1.0] range
        let value = (hash as f64 / u64::MAX as f64) * 2.0 - 1.0;
        embedding.push(value as f32);
    }

    embedding
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_embedding() {
        let data = b"AAPL,150.0,1000";

        let embed1 = hash_embed(data, 512);
        let embed2 = hash_embed(data, 512);

        // Should be identical
        assert_eq!(embed1, embed2);
    }

    #[test]
    fn test_embedding_uniqueness() {
        let data1 = b"AAPL,150.0,1000";
        let data2 = b"AAPL,150.1,1000";

        let embed1 = hash_embed(data1, 512);
        let embed2 = hash_embed(data2, 512);

        // Should be different
        assert_ne!(embed1, embed2);

        // Calculate cosine similarity
        let similarity = cosine_similarity(&embed1, &embed2);
        assert!(similarity < 0.99); // Not too similar
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    dot / (norm_a * norm_b)
}
```

### Advantages

✅ **Deterministic:** Same input always produces same embedding
✅ **Fast:** No neural network inference required
✅ **Collision-resistant:** Uses cryptographic hash function
✅ **Reversible:** Store original data separately for reconstruction

### Use Cases

- **Deduplication:** Detect duplicate market data
- **Similarity search:** Find similar market conditions
- **Pattern matching:** Match trading patterns
- **Anomaly detection:** Identify outliers

---

## Provenance Tracking with Ed25519

### Implementation

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
        let created_us = Utc::now().timestamp_micros();

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
    pub fn sign<T: Serialize>(mut self, data: &T, keypair: &Keypair) -> Result<Self> {
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
    pub fn verify(&self) -> Result<bool> {
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

/// Generate keypair for signing
pub fn generate_keypair() -> Keypair {
    let mut csprng = rand::rngs::OsRng;
    Keypair::generate(&mut csprng)
}
```

### Usage Example

```rust
// Create keypair
let keypair = generate_keypair();

// Create and sign observation
let mut obs = Observation::new("AAPL".to_string(), Decimal::new(15050, 2), Decimal::new(1000, 0));
obs.provenance = obs.provenance.sign(&obs, &keypair)?;

// Verify later
assert!(obs.provenance.verify()?);
```

---

## Query Templates with Performance Targets

### 1. Find Similar Market Conditions

**Use Case:** Given current market state, find similar historical states

```rust
pub async fn find_similar_conditions(
    db: &AgentDBClient,
    current: &Observation,
    k: usize,
    time_window_hours: Option<i64>,
) -> Result<Vec<Observation>> {
    let mut query = Query::new(&current.embedding)
        .collection("observations")
        .k(k)
        .filter(Filter::eq("symbol", &current.symbol));

    // Optional: Restrict to recent history
    if let Some(hours) = time_window_hours {
        let cutoff = current.timestamp_us - (hours * 3600 * 1_000_000);
        query = query.filter(Filter::gte("timestamp_us", cutoff));
    }

    let results = db.search(query).await?;
    Ok(results)
}
```

**Performance Target:** <1ms for k=10, <2ms for k=100

---

### 2. Get Signals by Strategy

**Use Case:** Retrieve all signals generated by a specific strategy

```rust
pub async fn get_signals_by_strategy(
    db: &AgentDBClient,
    strategy_id: &str,
    min_confidence: f64,
    limit: usize,
) -> Result<Vec<Signal>> {
    let query = Query::new_filter(
        Filter::and(vec![
            Filter::eq("strategy_id", strategy_id),
            Filter::gte("confidence", min_confidence),
        ])
    )
    .collection("signals")
    .limit(limit)
    .sort_by("confidence", SortOrder::Desc);

    let results = db.search(query).await?;
    Ok(results)
}
```

**Performance Target:** <1ms

---

### 3. Find Similar Trading Decisions

**Use Case:** Find similar past signals to learn from

```rust
pub async fn find_similar_decisions(
    db: &AgentDBClient,
    signal: &Signal,
    k: usize,
) -> Result<Vec<Signal>> {
    let query = Query::new(&signal.embedding)
        .collection("signals")
        .k(k)
        .filter(Filter::eq("symbol", &signal.symbol));

    let results = db.search(query).await?;
    Ok(results)
}
```

**Performance Target:** <1ms for k=10

---

### 4. Get Top Performing Strategies

**Use Case:** Rank strategies by performance

```rust
pub async fn get_top_strategies(
    db: &AgentDBClient,
    min_score: f64,
    limit: usize,
) -> Result<Vec<(String, f64)>> {
    let query = Query::new_filter(
        Filter::gte("verdict.score", min_score)
    )
    .collection("reflexion_traces")
    .limit(limit)
    .sort_by("verdict.sharpe", SortOrder::Desc);

    let traces: Vec<ReflexionTrace> = db.search(query).await?;

    // Aggregate by strategy
    let mut strategy_scores: HashMap<String, Vec<f64>> = HashMap::new();

    for trace in traces {
        // Extract strategy from decision
        strategy_scores.entry(extract_strategy_id(&trace))
            .or_default()
            .push(trace.verdict.score);
    }

    // Calculate average scores
    let mut results: Vec<(String, f64)> = strategy_scores
        .into_iter()
        .map(|(strategy, scores)| {
            let avg = scores.iter().sum::<f64>() / scores.len() as f64;
            (strategy, avg)
        })
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results.truncate(limit);

    Ok(results)
}
```

**Performance Target:** <50ms for 1000 traces

---

### 5. Temporal Range Query

**Use Case:** Get all observations in time window

```rust
pub async fn get_observations_in_range(
    db: &AgentDBClient,
    symbol: &str,
    start_us: i64,
    end_us: i64,
) -> Result<Vec<Observation>> {
    let query = Query::new_filter(
        Filter::and(vec![
            Filter::eq("symbol", symbol),
            Filter::gte("timestamp_us", start_us),
            Filter::lte("timestamp_us", end_us),
        ])
    )
    .collection("observations")
    .limit(10000)
    .sort_by("timestamp_us", SortOrder::Asc);

    let results = db.search(query).await?;
    Ok(results)
}
```

**Performance Target:** <5ms for 1-hour window, <50ms for 1-day window

---

## Reflexion Loop Integration

### ReasoningBank Pattern Implementation

```rust
use async_trait::async_trait;

#[async_trait]
pub trait ReflexionEngine: Send + Sync {
    async fn reflect(&self, decision_id: Uuid) -> Result<ReflexionTrace>;
    async fn learn_patterns(&self, trace: &ReflexionTrace) -> Result<Vec<Pattern>>;
    async fn apply_patterns(&self, signal: &mut Signal) -> Result<()>;
}

pub struct AgentDBReflexionEngine {
    db: Arc<AgentDBClient>,
    session: Arc<SessionMemory>,
}

impl AgentDBReflexionEngine {
    /// Reflect on past signal
    pub async fn reflect_on_signal(&self, signal_id: Uuid) -> Result<ReflexionTrace> {
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
        let mut trace = ReflexionTrace {
            id: Uuid::new_v4(),
            decision_id: signal_id,
            decision_type: DecisionType::Signal,
            trajectory,
            verdict,
            learned_patterns,
            counterfactuals,
            embedding: Vec::new(),
            provenance: Provenance::new("reflexion_engine"),
        };

        trace.embedding = trace.compute_embedding();

        self.db.insert(
            "reflexion_traces",
            trace.id.as_bytes(),
            &trace.embedding,
            Some(&trace)
        ).await?;

        Ok(trace)
    }

    /// Learn from similar past experiences
    pub async fn learn_from_history(&self, current_signal: &Signal) -> Result<Vec<Pattern>> {
        // Find similar past signals
        let similar_signals = find_similar_decisions(&self.db, current_signal, 10).await?;

        // Get reflexion traces for those signals
        let mut all_patterns = Vec::new();

        for signal in similar_signals {
            let traces = self.db.search(
                Query::new_filter(Filter::eq("decision_id", signal.id))
                    .collection("reflexion_traces")
            ).await?;

            for trace in traces {
                all_patterns.extend(trace.learned_patterns.clone());
            }
        }

        // Deduplicate and rank patterns
        let patterns = self.deduplicate_patterns(all_patterns);

        Ok(patterns)
    }

    /// Apply learned patterns to new signal
    pub async fn enhance_signal(&self, signal: &mut Signal) -> Result<()> {
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

### Learning Loop

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

## Memory Hierarchy

### Three-Tier Architecture

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

### L1: Session Memory (Hot Cache)

```rust
use dashmap::DashMap;
use parking_lot::RwLock;
use std::time::{Duration, Instant};

pub struct SessionMemory {
    /// Current positions by symbol
    positions: DashMap<String, Position>,

    /// Active orders by ID
    orders: DashMap<Uuid, Order>,

    /// Recent observations (ring buffer)
    observations: Arc<RwLock<RingBuffer<Observation>>>,

    /// Active signals
    signals: DashMap<Uuid, Signal>,

    /// Market state cache with TTL
    market_cache: Arc<RwLock<TtlCache<String, MarketState>>>,
}

impl SessionMemory {
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            positions: DashMap::new(),
            orders: DashMap::new(),
            observations: Arc::new(RwLock::new(RingBuffer::new(capacity))),
            signals: DashMap::new(),
            market_cache: Arc::new(RwLock::new(TtlCache::new(ttl))),
        }
    }

    /// Get current position for symbol (zero-copy)
    pub fn get_position(&self, symbol: &str) -> Option<Position> {
        self.positions.get(symbol).map(|p| p.clone())
    }

    /// Update position (< 100ns)
    pub fn update_position(&self, symbol: String, position: Position) {
        self.positions.insert(symbol, position);
    }

    /// Get recent observations for symbol (< 1μs)
    pub fn get_recent_observations(&self, symbol: &str, n: usize) -> Vec<Observation> {
        self.observations
            .read()
            .iter()
            .filter(|obs| obs.symbol == symbol)
            .take(n)
            .cloned()
            .collect()
    }

    /// Check if order is active (< 50ns)
    pub fn is_order_active(&self, order_id: &Uuid) -> bool {
        self.orders
            .get(order_id)
            .map(|o| matches!(o.status, OrderStatus::Pending | OrderStatus::Submitted))
            .unwrap_or(false)
    }
}
```

**Performance Targets:**
- Position lookup: <100ns
- Order status check: <50ns
- Recent observations: <1μs
- Cache hit rate: >95%

---

### L2: AgentDB Long-Term Memory

```rust
pub struct LongTermMemory {
    db: Arc<AgentDBClient>,
}

impl LongTermMemory {
    pub async fn new(endpoint: &str) -> Result<Self> {
        let db = AgentDBClient::connect(endpoint).await?;

        // Initialize collections
        db.create_collection(observation_collection_config()).await?;
        db.create_collection(signal_collection_config()).await?;
        db.create_collection(order_collection_config()).await?;
        db.create_collection(reflexion_collection_config()).await?;

        Ok(Self {
            db: Arc::new(db),
        })
    }

    /// Store observation (async, <5ms)
    pub async fn store_observation(&self, obs: &Observation) -> Result<()> {
        self.db.insert(
            "observations",
            obs.id.as_bytes(),
            &obs.embedding,
            Some(obs)
        ).await?;
        Ok(())
    }

    /// Find similar market conditions (< 1ms target)
    pub async fn find_similar_conditions(
        &self,
        query_obs: &Observation,
        k: usize,
        time_range: Option<(i64, i64)>,
    ) -> Result<Vec<Observation>> {
        find_similar_conditions(&self.db, query_obs, k, time_range.map(|(start, _)| {
            (query_obs.timestamp_us - start) / 3600_000_000
        })).await
    }
}
```

**Performance Targets:**
- Vector search: <1ms
- Filtered search: <2ms
- Batch insert: <10ms (1000 items)
- Storage overhead: 4-32x reduction via quantization

---

## Rust Client Implementation

### Complete Example

```rust
// main.rs
use neural_trader::{
    memory::{SessionMemory, LongTermMemory, AgentDBReflexionEngine},
    schema::{Observation, Signal},
};
use std::time::Duration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize memory layers
    let session = SessionMemory::new(10000, Duration::from_secs(300));
    let longterm = LongTermMemory::new("http://localhost:8080").await?;
    let reflexion = AgentDBReflexionEngine::new(
        longterm.db.clone(),
        Arc::new(session.clone())
    );

    // Create observation
    let obs = Observation::new(
        "AAPL".to_string(),
        Decimal::new(17543, 2),  // 175.43
        Decimal::new(100000, 0),
    );

    // Store in both layers
    session.store_observation(&obs);
    longterm.store_observation(&obs).await?;

    // Find similar conditions
    let similar = longterm.find_similar_conditions(&obs, 10, None).await?;
    println!("Found {} similar conditions", similar.len());

    // Generate signal
    let mut signal = generate_signal(&obs)?;

    // Enhance with learned patterns
    reflexion.enhance_signal(&mut signal).await?;

    // Execute and reflect
    execute_signal(&signal).await?;
    let trace = reflexion.reflect_on_signal(signal.id).await?;

    println!("Reflection complete. Score: {}", trace.verdict.score);

    Ok(())
}
```

---

## Cross-References

- **Architecture:** [03_Architecture.md](03_Architecture.md)
- **Parity Requirements:** [02_Parity_Requirements.md](02_Parity_Requirements.md)
- **Strategy Implementation:** [06_Strategy_and_Sublinear_Solvers.md](06_Strategy_and_Sublinear_Solvers.md)
- **AgentDB Full Docs:** [../../docs/RUST_AGENTDB_MEMORY_ARCHITECTURE.md](../../docs/RUST_AGENTDB_MEMORY_ARCHITECTURE.md)

---

**Document Status:** ✅ Complete
**Last Updated:** 2025-11-12
**Next Review:** Phase 3 (Week 7)
**Owner:** ML Developer + Backend Developer
