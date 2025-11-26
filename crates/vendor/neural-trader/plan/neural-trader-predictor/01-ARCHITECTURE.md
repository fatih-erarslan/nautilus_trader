# Neural Trader Predictor - Architecture Design

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Applications                         │
│  (Trading Bots, Analysis Tools, Dashboards)                 │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
┌───────────▼──────────┐       ┌──────────▼───────────┐
│   Rust CLI Tool      │       │  JS/WASM/NAPI API    │
│  (Standalone)        │       │  (@neural-trader/    │
│                      │       │   predictor)         │
└───────────┬──────────┘       └──────────┬───────────┘
            │                             │
            └───────────┬─────────────────┘
                        │
        ┌───────────────▼───────────────┐
        │   Rust Core Library           │
        │   (neural-trader-predictor)   │
        │                               │
        │  ┌─────────────────────────┐  │
        │  │ Conformal Prediction    │  │
        │  │  - Split CP             │  │
        │  │  - Adaptive CI (ACI)    │  │
        │  │  - CQR                  │  │
        │  └─────────────────────────┘  │
        │                               │
        │  ┌─────────────────────────┐  │
        │  │ Nonconformity Scores    │  │
        │  │  - Absolute             │  │
        │  │  - Normalized           │  │
        │  │  - Quantile (CQR)       │  │
        │  └─────────────────────────┘  │
        │                               │
        │  ┌─────────────────────────┐  │
        │  │ Performance Optimizers  │  │
        │  │  - Nanosecond Scheduler │  │
        │  │  - Sublinear Algorithms │  │
        │  │  - Temporal Lead Solver │  │
        │  │  - Strange Loops        │  │
        │  └─────────────────────────┘  │
        └───────────────────────────────┘
                        │
        ┌───────────────▼───────────────┐
        │   Base Model Interface        │
        │  (Neural Networks, XGBoost)   │
        └───────────────────────────────┘
```

## 📦 Crate Structure

### Rust Crate: `neural-trader-predictor`

```
neural-trader-predictor/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # Public API exports
│   ├── core/
│   │   ├── mod.rs
│   │   ├── types.rs              # PredictionInterval, Result types
│   │   ├── errors.rs             # Error definitions
│   │   └── traits.rs             # Score, Predictor traits
│   ├── conformal/
│   │   ├── mod.rs
│   │   ├── split.rs              # Split conformal prediction
│   │   ├── adaptive.rs           # Adaptive conformal inference (ACI)
│   │   └── cqr.rs                # Conformalized quantile regression
│   ├── scores/
│   │   ├── mod.rs
│   │   ├── absolute.rs           # Absolute residual score
│   │   ├── normalized.rs         # Normalized score
│   │   └── quantile.rs           # Quantile-based score
│   ├── optimizers/
│   │   ├── mod.rs
│   │   ├── scheduler.rs          # Nanosecond scheduler integration
│   │   ├── sublinear.rs          # Sublinear update algorithms
│   │   ├── temporal.rs           # Temporal lead solver
│   │   └── loops.rs              # Strange loops optimization
│   ├── cli/
│   │   ├── mod.rs
│   │   ├── commands.rs           # CLI command definitions
│   │   └── config.rs             # Configuration parsing
│   └── bin/
│       └── neural-predictor.rs   # CLI binary entry point
├── tests/
│   ├── integration_tests.rs
│   ├── conformal_tests.rs
│   └── benchmark_tests.rs
├── benches/
│   ├── prediction_bench.rs
│   └── calibration_bench.rs
└── examples/
    ├── basic_usage.rs
    ├── adaptive_trading.rs
    └── quantile_regression.rs
```

### NPM Package: `@neural-trader/predictor`

```
packages/predictor/
├── package.json
├── Cargo.toml                    # For WASM build
├── src/
│   ├── index.ts                  # Main entry point
│   ├── pure/                     # Pure JS implementation
│   │   ├── conformal.ts
│   │   ├── scores.ts
│   │   └── types.ts
│   ├── wasm/                     # WASM bindings
│   │   ├── lib.rs               # Rust WASM interface
│   │   └── index.ts             # WASM loader
│   └── napi/                     # Optional native bindings
│       ├── lib.rs               # NAPI-rs interface
│       └── index.ts             # Native loader
├── tests/
│   ├── conformal.test.ts
│   ├── wasm.test.ts
│   └── benchmark.test.ts
├── benchmarks/
│   └── performance.bench.ts
└── examples/
    ├── basic.ts
    ├── trading.ts
    └── streaming.ts
```

## 🔑 Core Data Structures

### PredictionInterval

```rust
pub struct PredictionInterval {
    /// Point prediction from base model
    pub point: f64,

    /// Lower bound of prediction interval
    pub lower: f64,

    /// Upper bound of prediction interval
    pub upper: f64,

    /// Miscoverage rate (1 - coverage)
    pub alpha: f64,

    /// Computed quantile threshold
    pub quantile: f64,

    /// Timestamp of prediction
    pub timestamp: i64,
}

impl PredictionInterval {
    pub fn width(&self) -> f64;
    pub fn contains(&self, value: f64) -> bool;
    pub fn relative_width(&self) -> f64;
}
```

### SplitConformalPredictor

```rust
pub struct SplitConformalPredictor<S: NonconformityScore> {
    /// Sorted calibration scores
    calibration_scores: Vec<f64>,

    /// Nonconformity score function
    score_fn: S,

    /// Number of calibration samples
    n_calibration: usize,

    /// Target miscoverage rate
    alpha: f64,

    /// Computed quantile
    quantile: f64,
}

impl<S: NonconformityScore> SplitConformalPredictor<S> {
    pub fn new(alpha: f64, score_fn: S) -> Self;
    pub fn calibrate(&mut self, predictions: &[f64], actuals: &[f64]) -> Result<()>;
    pub fn predict(&self, point_prediction: f64) -> PredictionInterval;
    pub fn update(&mut self, prediction: f64, actual: f64) -> Result<()>;
}
```

### AdaptiveConformalPredictor

```rust
pub struct AdaptiveConformalPredictor<S: NonconformityScore> {
    /// Base conformal predictor
    base: SplitConformalPredictor<S>,

    /// Adaptive alpha (PID control)
    alpha_current: f64,

    /// Target coverage
    target_coverage: f64,

    /// Learning rate (gamma)
    gamma: f64,

    /// Coverage history for monitoring
    coverage_history: VecDeque<f64>,
}

impl<S: NonconformityScore> AdaptiveConformalPredictor<S> {
    pub fn new(target_coverage: f64, gamma: f64, score_fn: S) -> Self;
    pub fn predict_and_adapt(&mut self, point: f64, actual: Option<f64>) -> PredictionInterval;
    pub fn empirical_coverage(&self) -> f64;
}
```

## 🔌 Trait System

### NonconformityScore Trait

```rust
pub trait NonconformityScore: Send + Sync {
    /// Compute nonconformity score
    fn score(&self, prediction: f64, actual: f64) -> f64;

    /// Optional: Compute prediction interval given quantile
    fn interval(&self, prediction: f64, quantile: f64) -> (f64, f64) {
        (prediction - quantile, prediction + quantile)
    }
}
```

### BaseModel Trait

```rust
pub trait BaseModel: Send + Sync {
    /// Make a point prediction
    fn predict(&self, features: &[f64]) -> Result<f64>;

    /// Optional: Batch predictions
    fn predict_batch(&self, features: &[Vec<f64>]) -> Result<Vec<f64>>;
}
```

## 🚀 Performance Optimization Strategy

### 1. Nanosecond Scheduler
- Schedule calibration updates during market idle periods
- Prioritize real-time predictions over background tasks
- Sub-microsecond task dispatch

### 2. Sublinear Algorithms
- Binary search for score insertion: O(log n)
- Incremental quantile updates
- Lazy recalibration triggers

### 3. Temporal Lead Solver
- Pre-compute next interval before features arrive
- Predictive calibration based on historical patterns
- Speculative execution for hot paths

### 4. Strange Loops
- Recursive optimization of prediction pipelines
- Self-tuning gamma parameters
- Meta-learning for alpha adjustment

## 🔗 Integration Points

### With @neural-trader/neural
```typescript
import { NeuralPredictor } from '@neural-trader/neural';
import { ConformalWrapper } from '@neural-trader/predictor';

const neural = new NeuralPredictor(modelPath);
const conformal = new ConformalWrapper(neural, { alpha: 0.1 });

const interval = await conformal.predictInterval(features);
if (interval.width < maxWidth) {
    executeTrade(interval);
}
```

### CLI Interface
```bash
# Calibrate model
neural-predictor calibrate \
    --model-path ./model.onnx \
    --calibration-data ./calib.csv \
    --alpha 0.1 \
    --output ./predictor.json

# Make predictions
neural-predictor predict \
    --predictor ./predictor.json \
    --features "1.2,3.4,5.6" \
    --format json

# Adaptive mode
neural-predictor stream \
    --predictor ./predictor.json \
    --input-stream tcp://localhost:9000 \
    --adaptive \
    --gamma 0.02
```

## 📊 Monitoring & Observability

### Metrics to Track
- Empirical coverage rate
- Average interval width
- Prediction latency (p50, p95, p99)
- Calibration drift
- Alpha adjustment rate (adaptive mode)

### Health Checks
- Coverage within 2% of target
- No calibration samples older than threshold
- Interval width distribution reasonable
- No prediction latency spikes

## 🧪 Testing Strategy

### Unit Tests
- Conformal prediction correctness
- Nonconformity score calculations
- Quantile computation accuracy
- Adaptive alpha adjustments

### Integration Tests
- End-to-end prediction pipeline
- Calibration with real market data
- WASM/NAPI bindings correctness
- CLI command execution

### Property Tests (proptest)
- Coverage guarantee holds for random data
- Interval monotonicity with alpha
- Calibration convergence

### Benchmarks
- Prediction latency vs. calibration size
- Memory usage scaling
- Comparison: Rust vs. WASM vs. Pure JS vs. NAPI
- Comparison: Conformal vs. Bootstrap vs. MC Dropout
