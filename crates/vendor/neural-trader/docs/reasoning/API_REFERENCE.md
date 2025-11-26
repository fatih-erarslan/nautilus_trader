# ReasoningBank API Reference

Complete API documentation for the ReasoningBank self-learning engine.

## Table of Contents

- [PatternLearningEngine](#patternlearningengine)
- [Types](#types)
- [Metrics](#metrics)
- [Traits](#traits)

---

## PatternLearningEngine

The main self-learning system for pattern recognition and adaptation.

### Constructor

```rust
pub fn new(storage: Arc<dyn PatternStorage>) -> Self
```

Create a new learning engine with the specified storage backend.

**Parameters:**
- `storage` - Implementation of `PatternStorage` trait (typically AgentDB)

**Returns:** New `PatternLearningEngine` instance

**Example:**
```rust
let storage = Arc::new(MyStorage::new());
let engine = PatternLearningEngine::new(storage);
```

### Methods

#### record_experience

```rust
pub async fn record_experience(&self, experience: PatternExperience) -> Result<()>
```

Record a new pattern matching experience.

**Parameters:**
- `experience` - The pattern experience to record

**Returns:** `Result<()>`

**Behavior:**
- Adds to in-memory cache
- Stores in persistent storage with vector embedding
- Trims cache if exceeds `max_memory_size`

---

#### update_outcome

```rust
pub async fn update_outcome(
    &self,
    experience_id: &str,
    actual_outcome: f64,
) -> Result<PatternVerdict>
```

Update an experience with actual outcome and generate verdict.

**Parameters:**
- `experience_id` - ID of the experience to update
- `actual_outcome` - The realized return

**Returns:** `Result<PatternVerdict>` with quality assessment

**Behavior:**
- Updates experience in memory
- Judges prediction quality
- Distills to memory if quality > 0.8
- Triggers adaptation if quality < 0.4

---

#### build_trajectory

```rust
pub async fn build_trajectory(
    &self,
    pattern_type: &str,
) -> Result<PatternTrajectory>
```

Build performance trajectory for a pattern type.

**Parameters:**
- `pattern_type` - Type of pattern to analyze

**Returns:** `Result<PatternTrajectory>` with performance metrics

**Errors:**
- Returns error if no completed experiences found

**Metrics Calculated:**
- Sample count
- Success rate (win rate)
- Average return
- Sharpe ratio
- Best/worst returns

---

#### adapt_thresholds

```rust
pub async fn adapt_thresholds(
    &self,
    lookback: usize,
) -> Result<MatchingThresholds>
```

Adaptively adjust matching thresholds based on recent performance.

**Parameters:**
- `lookback` - Number of recent experiences to analyze

**Returns:** `Result<MatchingThresholds>` with updated thresholds

**Adaptation Logic:**
- **Poor performance** (win rate < 55%, PF < 1.2): +0.03 to thresholds
- **Good performance** (win rate > 70%, PF > 1.8): -0.02 to thresholds
- **Medium performance**: No change

**Minimum Samples:** Requires at least 10 completed experiences

---

#### get_thresholds

```rust
pub async fn get_thresholds(&self) -> MatchingThresholds
```

Get current matching thresholds.

**Returns:** Current `MatchingThresholds`

---

#### get_experiences

```rust
pub async fn get_experiences(&self) -> Vec<PatternExperience>
```

Get all cached experiences.

**Returns:** Vector of all `PatternExperience` in memory cache

---

#### get_trajectory

```rust
pub async fn get_trajectory(
    &self,
    pattern_type: &str,
) -> Option<PatternTrajectory>
```

Get cached trajectory for a pattern type.

**Parameters:**
- `pattern_type` - Type of pattern

**Returns:** `Option<PatternTrajectory>` (None if not cached)

---

## Types

### PatternExperience

A recorded pattern matching experience.

```rust
pub struct PatternExperience {
    pub id: String,
    pub pattern_type: String,
    pub pattern_vector: Vec<f32>,
    pub similarity: f64,
    pub confidence: f64,
    pub predicted_outcome: f64,
    pub actual_outcome: Option<f64>,
    pub market_context: MarketContext,
    pub timestamp: DateTime<Utc>,
}
```

**Fields:**
- `id` - Unique identifier
- `pattern_type` - Type of pattern (e.g., "head_and_shoulders")
- `pattern_vector` - Vector embedding for similarity search
- `similarity` - Similarity score to historical pattern (0-1)
- `confidence` - Confidence in prediction (0-1)
- `predicted_outcome` - Expected return
- `actual_outcome` - Actual return (filled later)
- `market_context` - Market conditions
- `timestamp` - When recorded

---

### MarketContext

Market conditions when pattern detected.

```rust
pub struct MarketContext {
    pub symbol: String,
    pub timeframe: String,
    pub volatility: f64,
    pub volume: f64,
    pub trend: String,
    pub sentiment: f64,
}
```

**Fields:**
- `symbol` - Trading symbol (e.g., "BTC-USD")
- `timeframe` - Time period (e.g., "1h", "4h")
- `volatility` - Market volatility (standard deviation)
- `volume` - Trading volume
- `trend` - Market trend ("bullish", "bearish", "neutral")
- `sentiment` - Sentiment score (-1 to 1)

---

### PatternVerdict

Verdict on prediction quality.

```rust
pub struct PatternVerdict {
    pub experience_id: String,
    pub quality_score: f64,
    pub direction_correct: bool,
    pub magnitude_error: f64,
    pub should_learn: bool,
    pub should_adapt: bool,
    pub suggested_changes: Vec<Adaptation>,
}
```

**Fields:**
- `experience_id` - Reference to experience
- `quality_score` - Overall quality (0-1)
- `direction_correct` - Whether direction prediction was correct
- `magnitude_error` - Magnitude prediction error (0-1)
- `should_learn` - Whether to distill to memory (score > 0.7)
- `should_adapt` - Whether to adapt thresholds (score < 0.4)
- `suggested_changes` - Recommended parameter adjustments

**Quality Score Ranges:**
- 0.9-1.0: Excellent
- 0.8-0.9: Good
- 0.7-0.8: Acceptable
- 0.5-0.7: Poor
- 0.0-0.5: Failed

---

### Adaptation

Suggested parameter adaptation.

```rust
pub struct Adaptation {
    pub parameter: String,
    pub current_value: f64,
    pub suggested_value: f64,
    pub reason: String,
}
```

**Fields:**
- `parameter` - Parameter name (e.g., "similarity_threshold")
- `current_value` - Current value
- `suggested_value` - Recommended new value
- `reason` - Explanation for suggestion

---

### DistilledPattern

Successful pattern in long-term memory.

```rust
pub struct DistilledPattern {
    pub pattern_type: String,
    pub pattern_vector: Vec<f32>,
    pub success_rate: f64,
    pub avg_return: f64,
    pub confidence_threshold: f64,
    pub similarity_threshold: f64,
    pub market_conditions: MarketContext,
    pub sample_count: usize,
    pub last_updated: DateTime<Utc>,
}
```

**Fields:**
- `pattern_type` - Type of pattern
- `pattern_vector` - Vector embedding
- `success_rate` - Historical success rate (0-1)
- `avg_return` - Average return
- `confidence_threshold` - Minimum confidence for this pattern
- `similarity_threshold` - Minimum similarity for this pattern
- `market_conditions` - Typical market context
- `sample_count` - Number of successful instances
- `last_updated` - Last update time

---

### PatternTrajectory

Performance trajectory over time.

```rust
pub struct PatternTrajectory {
    pub pattern_type: String,
    pub sample_count: usize,
    pub success_rate: f64,
    pub avg_return: f64,
    pub sharpe_ratio: f64,
    pub best_return: f64,
    pub worst_return: f64,
    pub experiences: Vec<PatternExperience>,
    pub last_updated: DateTime<Utc>,
}
```

**Fields:**
- `pattern_type` - Pattern being tracked
- `sample_count` - Total samples
- `success_rate` - Win rate (profitable / total)
- `avg_return` - Mean return
- `sharpe_ratio` - Risk-adjusted return (annualized)
- `best_return` - Best return achieved
- `worst_return` - Worst return (max loss)
- `experiences` - All experiences for this pattern
- `last_updated` - Last trajectory update

---

### MatchingThresholds

Pattern matching thresholds.

```rust
pub struct MatchingThresholds {
    pub similarity_threshold: f64,
    pub confidence_threshold: f64,
    pub min_sample_size: usize,
}
```

**Fields:**
- `similarity_threshold` - Minimum similarity to match (0-1)
- `confidence_threshold` - Minimum confidence to signal (0-1)
- `min_sample_size` - Minimum historical samples required

**Defaults:**
- similarity_threshold: 0.80
- confidence_threshold: 0.70
- min_sample_size: 10

---

## Metrics

Financial metrics for performance evaluation.

### calculate_sharpe_ratio

```rust
pub fn calculate_sharpe_ratio(returns: &[f64]) -> f64
```

Calculate annualized Sharpe ratio.

**Parameters:**
- `returns` - Slice of return values

**Returns:** Annualized Sharpe ratio (assumes daily returns, 252 trading days)

**Risk-free rate:** 2% (default)

---

### calculate_sharpe_ratio_with_rf

```rust
pub fn calculate_sharpe_ratio_with_rf(
    returns: &[f64],
    risk_free_rate: f64,
) -> f64
```

Calculate Sharpe ratio with custom risk-free rate.

---

### calculate_sortino_ratio

```rust
pub fn calculate_sortino_ratio(returns: &[f64]) -> f64
```

Calculate annualized Sortino ratio (downside deviation only).

**Target return:** 0.0 (default)

---

### calculate_max_drawdown

```rust
pub fn calculate_max_drawdown(returns: &[f64]) -> f64
```

Calculate maximum drawdown.

**Returns:** Max drawdown as positive percentage (0-1)

---

### calculate_calmar_ratio

```rust
pub fn calculate_calmar_ratio(returns: &[f64]) -> f64
```

Calculate Calmar ratio (annualized return / max drawdown).

---

### calculate_win_rate

```rust
pub fn calculate_win_rate(returns: &[f64]) -> f64
```

Calculate win rate (percentage of profitable trades).

**Returns:** Win rate (0-1)

---

### calculate_profit_factor

```rust
pub fn calculate_profit_factor(returns: &[f64]) -> f64
```

Calculate profit factor (gross profit / gross loss).

**Returns:** Profit factor (> 1.0 is profitable)

---

### calculate_var

```rust
pub fn calculate_var(returns: &[f64], confidence: f64) -> f64
```

Calculate Value at Risk at confidence level.

**Parameters:**
- `returns` - Return values
- `confidence` - Confidence level (e.g., 0.95 for 95%)

**Returns:** VaR as positive value

---

### calculate_cvar

```rust
pub fn calculate_cvar(returns: &[f64], confidence: f64) -> f64
```

Calculate Conditional VaR (expected loss beyond VaR).

---

## Traits

### PatternStorage

Storage backend for pattern data.

```rust
#[async_trait]
pub trait PatternStorage: Send + Sync {
    async fn insert(
        &self,
        collection: &str,
        data: &JsonValue,
        vector: Option<&[f32]>,
    ) -> Result<String>;

    async fn query(
        &self,
        collection: &str,
        filter: &str,
        limit: usize,
    ) -> Result<Vec<JsonValue>>;

    async fn query_similar(
        &self,
        collection: &str,
        vector: &[f32],
        limit: usize,
    ) -> Result<Vec<JsonValue>>;

    async fn update(
        &self,
        collection: &str,
        id: &str,
        data: &JsonValue,
    ) -> Result<()>;
}
```

**Methods:**
- `insert` - Store pattern with optional vector embedding
- `query` - Query patterns with filter
- `query_similar` - Vector similarity search
- `update` - Update existing pattern

---

## Error Handling

All async methods return `anyhow::Result<T>`.

**Common Errors:**
- Experience not found (invalid ID)
- No completed experiences (trajectory building)
- Insufficient data for adaptation (< 10 samples)
- Storage backend errors

**Example:**
```rust
match engine.update_outcome("exp_001", 0.05).await {
    Ok(verdict) => println!("Quality: {:.2}", verdict.quality_score),
    Err(e) => eprintln!("Error: {}", e),
}
```

---

## Thread Safety

All types are thread-safe:
- `PatternLearningEngine` uses `Arc<RwLock<T>>` internally
- Safe to share across threads
- Multiple readers or single writer

**Example:**
```rust
let engine = Arc::new(PatternLearningEngine::new(storage));
let engine_clone = Arc::clone(&engine);

tokio::spawn(async move {
    engine_clone.record_experience(exp).await
});
```

---

## Performance Characteristics

### Time Complexity
- `record_experience`: O(1) + storage write
- `update_outcome`: O(n) where n = cache size
- `build_trajectory`: O(n) where n = pattern samples
- `adapt_thresholds`: O(n) where n = lookback

### Space Complexity
- In-memory cache: O(max_memory_size) (default 1000)
- Storage: O(total experiences)

### Concurrency
- Multiple concurrent reads: ✅
- Concurrent read/write: ✅ (RwLock)
- Multiple concurrent writes: ⚠️ (serialized by RwLock)

---

## Examples

See [Quick Start Guide](./QUICK_START.md) for complete examples.
