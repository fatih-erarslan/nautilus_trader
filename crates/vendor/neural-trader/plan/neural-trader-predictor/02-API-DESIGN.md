# Neural Trader Predictor - API Design

## 🦀 Rust API

### Core Library API

```rust
// Main entry point
use neural_trader_predictor::{
    ConformalPredictor,
    AdaptiveConformalPredictor,
    PredictionInterval,
    scores::{AbsoluteScore, NormalizedScore, QuantileScore},
    Result,
};

// Example 1: Basic Split Conformal Prediction
fn basic_usage() -> Result<()> {
    let mut predictor = ConformalPredictor::new(0.1, AbsoluteScore);

    // Calibrate with historical data
    let predictions = vec![100.0, 105.0, 98.0, 102.0];
    let actuals = vec![102.0, 104.0, 99.0, 101.0];
    predictor.calibrate(&predictions, &actuals)?;

    // Make prediction with interval
    let interval = predictor.predict(103.0);
    println!("Prediction: {} [{}, {}]",
             interval.point, interval.lower, interval.upper);

    // Update with new observation
    predictor.update(103.0, 102.5)?;

    Ok(())
}

// Example 2: Adaptive Conformal Inference
fn adaptive_usage() -> Result<()> {
    let mut predictor = AdaptiveConformalPredictor::new(
        0.90, // target 90% coverage
        0.02, // gamma learning rate
        AbsoluteScore,
    );

    // Streaming predictions with adaptation
    for (point_pred, actual) in stream_predictions() {
        let interval = predictor.predict_and_adapt(point_pred, Some(actual));

        if interval.width() < max_width {
            execute_trade(&interval);
        }

        println!("Coverage: {:.2}%", predictor.empirical_coverage() * 100.0);
    }

    Ok(())
}

// Example 3: Conformalized Quantile Regression
fn cqr_usage() -> Result<()> {
    let mut predictor = ConformalPredictor::new(0.1, QuantileScore::new(0.05, 0.95));

    // Calibrate with quantile predictions
    let q_low = vec![95.0, 100.0, 93.0, 97.0];
    let q_high = vec![105.0, 110.0, 103.0, 107.0];
    let actuals = vec![102.0, 104.0, 99.0, 101.0];

    predictor.calibrate_quantiles(&q_low, &q_high, &actuals)?;

    // Predict with adjusted quantiles
    let interval = predictor.predict_quantile(98.0, 108.0);

    Ok(())
}
```

### Configuration API

```rust
use neural_trader_predictor::config::{PredictorConfig, AdaptiveConfig};

let config = PredictorConfig {
    alpha: 0.1,
    calibration_size: 2000,
    score_type: "absolute",
    max_interval_width_pct: 5.0,
    recalibration_freq: 100,
};

let adaptive_config = AdaptiveConfig {
    target_coverage: 0.90,
    gamma: 0.02,
    coverage_window: 200,
    alpha_min: 0.01,
    alpha_max: 0.30,
};

let predictor = config.build()?;
```

### Optimizer Integration API

```rust
use neural_trader_predictor::optimizers::{
    NanosecondScheduler,
    SublinearUpdater,
    TemporalLeadSolver,
};

// Nanosecond-precision scheduling
let mut scheduler = NanosecondScheduler::new();
scheduler.schedule_recalibration(
    Duration::from_secs(60),
    Box::new(move || predictor.recalibrate())
);

// Sublinear updates for streaming
let mut updater = SublinearUpdater::new(predictor);
updater.stream_update(prediction, actual)?; // O(log n)

// Temporal lead solving (predictive pre-computation)
let mut temporal = TemporalLeadSolver::new(predictor);
let future_interval = temporal.solve_ahead(features, lead_time_ms)?;
```

### CLI API

```rust
// CLI command structure
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "neural-predictor")]
#[command(about = "Conformal prediction for neural trading")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Calibrate a conformal predictor
    Calibrate {
        #[arg(short, long)]
        model_path: String,

        #[arg(short, long)]
        calibration_data: String,

        #[arg(short, long, default_value = "0.1")]
        alpha: f64,

        #[arg(short, long)]
        output: String,
    },

    /// Make predictions with intervals
    Predict {
        #[arg(short, long)]
        predictor: String,

        #[arg(short, long)]
        features: String,

        #[arg(short, long, default_value = "json")]
        format: String,
    },

    /// Stream predictions (adaptive mode)
    Stream {
        #[arg(short, long)]
        predictor: String,

        #[arg(short, long)]
        input_stream: String,

        #[arg(long)]
        adaptive: bool,

        #[arg(short, long, default_value = "0.02")]
        gamma: f64,
    },

    /// Evaluate coverage on test data
    Evaluate {
        #[arg(short, long)]
        predictor: String,

        #[arg(short, long)]
        test_data: String,
    },

    /// Benchmark performance
    Benchmark {
        #[arg(short, long)]
        predictor: String,

        #[arg(short, long, default_value = "1000")]
        iterations: usize,
    },
}
```

## 🌐 JavaScript/TypeScript API

### Pure JS API

```typescript
import { ConformalPredictor, AbsoluteScore, PredictionInterval } from '@neural-trader/predictor';

// Basic usage
const predictor = new ConformalPredictor({
    alpha: 0.1,
    scoreFunction: new AbsoluteScore(),
});

// Calibrate
await predictor.calibrate(
    predictions: [100, 105, 98, 102],
    actuals: [102, 104, 99, 101]
);

// Predict
const interval = predictor.predict(103.0);
console.log(`Prediction: ${interval.point} [${interval.lower}, ${interval.upper}]`);
console.log(`Width: ${interval.width()}`);

// Update
await predictor.update(103.0, 102.5);
```

### Adaptive API

```typescript
import { AdaptiveConformalPredictor } from '@neural-trader/predictor';

const predictor = new AdaptiveConformalPredictor({
    targetCoverage: 0.90,
    gamma: 0.02,
    scoreFunction: new AbsoluteScore(),
});

// Streaming with adaptation
for await (const { prediction, actual } of marketStream) {
    const interval = await predictor.predictAndAdapt(prediction, actual);

    if (interval.width() < maxWidth && interval.point > threshold) {
        await executeTrade(interval);
    }

    console.log(`Coverage: ${predictor.empiricalCoverage() * 100}%`);
}
```

### WASM API (High Performance)

```typescript
import { WasmConformalPredictor, initWasm } from '@neural-trader/predictor/wasm';

// Initialize WASM module
await initWasm();

// Use WASM-accelerated predictor (same API as pure JS)
const predictor = new WasmConformalPredictor({
    alpha: 0.1,
    scoreFunction: 'absolute', // Runs in Rust
});

await predictor.calibrate(predictions, actuals);
const interval = predictor.predict(103.0);
// 5-10x faster than pure JS
```

### NAPI Native API (Maximum Performance)

```typescript
import { NativeConformalPredictor } from '@neural-trader/predictor/native';

// Native addon via NAPI-rs (fastest option)
const predictor = new NativeConformalPredictor({
    alpha: 0.1,
    scoreFunction: 'absolute',
});

await predictor.calibrate(predictions, actuals);
const interval = predictor.predict(103.0);
// Near-Rust performance in Node.js
```

### Factory Pattern (Auto-select best implementation)

```typescript
import { createPredictor } from '@neural-trader/predictor';

// Automatically selects: Native > WASM > Pure JS
const predictor = await createPredictor({
    alpha: 0.1,
    preferNative: true, // Try native first
    fallbackToWasm: true, // Then WASM
});

console.log(`Using: ${predictor.implementation}`); // "native" | "wasm" | "pure"
```

### Integration with @neural-trader/neural

```typescript
import { NeuralPredictor } from '@neural-trader/neural';
import { wrapWithConformal } from '@neural-trader/predictor';

const neural = new NeuralPredictor({ modelPath: './model.onnx' });

// Wrap neural predictor with conformal intervals
const conformal = wrapWithConformal(neural, {
    alpha: 0.1,
    calibrationSize: 2000,
    adaptive: true,
    gamma: 0.02,
});

// Now neural predictor returns intervals
const result = await conformal.predict(features);
console.log(`Interval: [${result.lower}, ${result.upper}]`);
console.log(`Confidence: ${result.confidence}%`);

if (result.shouldTrade(maxWidth: 0.05)) {
    await executeTrade(result);
}
```

### Trading Decision API

```typescript
import { TradingDecisionEngine } from '@neural-trader/predictor';

const engine = new TradingDecisionEngine({
    predictor: conformalPredictor,
    maxIntervalWidthPct: 5.0,
    minConfidence: 0.85,
    kellyFraction: 0.25,
});

const decision = await engine.evaluate(features);

if (decision.shouldTrade) {
    console.log(`Signal: ${decision.signal}`); // "buy" | "sell" | "hold"
    console.log(`Size: ${decision.positionSize}`);
    console.log(`Edge: ${decision.edge}%`);
    console.log(`Risk: ${decision.risk}%`);
}
```

### Monitoring API

```typescript
import { PredictorMonitor } from '@neural-trader/predictor';

const monitor = new PredictorMonitor(predictor);

// Real-time metrics
setInterval(() => {
    const metrics = monitor.getMetrics();
    console.log(`
        Coverage: ${metrics.empiricalCoverage}%
        Avg Width: ${metrics.avgIntervalWidth}
        Latency p95: ${metrics.latencyP95}ms
        Calibration Age: ${metrics.calibrationAge}s
    `);

    // Health check
    if (!monitor.isHealthy()) {
        console.error('Predictor unhealthy:', monitor.getIssues());
    }
}, 5000);
```

## 🔌 REST API (Optional Server Mode)

```typescript
import { PredictorServer } from '@neural-trader/predictor/server';

const server = new PredictorServer({
    port: 8080,
    predictor: conformalPredictor,
    auth: { apiKey: process.env.API_KEY },
});

await server.start();

// Endpoints:
// POST /api/predict
//   Body: { features: number[] }
//   Response: { point, lower, upper, alpha, quantile }

// POST /api/calibrate
//   Body: { predictions: number[], actuals: number[] }
//   Response: { success: true, calibrationSize: number }

// GET /api/metrics
//   Response: { coverage, avgWidth, latency, health }

// POST /api/update
//   Body: { prediction: number, actual: number }
//   Response: { success: true }
```

## 📊 Configuration Schema

```typescript
interface PredictorConfig {
    // Core parameters
    alpha: number;                    // 0.01-0.30 (default: 0.1)
    scoreFunction: ScoreFunction;     // absolute | normalized | quantile

    // Calibration
    calibrationSize?: number;         // 1000-5000 (default: 2000)
    recalibrationFreq?: number;       // predictions before recalib (default: 100)

    // Adaptive mode
    adaptive?: boolean;               // Enable ACI (default: false)
    targetCoverage?: number;          // 0.80-0.99 (default: 0.90)
    gamma?: number;                   // 0.01-0.05 (default: 0.02)

    // Trading constraints
    maxIntervalWidthPct?: number;     // 1-10% (default: 5.0)
    minConfidence?: number;           // 0.50-0.99 (default: 0.80)

    // Performance
    implementation?: 'auto' | 'native' | 'wasm' | 'pure';
    parallelism?: number;             // Worker threads (default: 4)

    // Monitoring
    monitoring?: {
        enabled: boolean;
        metricsInterval: number;      // ms (default: 5000)
        healthCheckInterval: number;  // ms (default: 30000)
    };
}
```

## 🧪 Testing Utilities API

```typescript
import {
    generateSyntheticData,
    evaluateCoverage,
    compareMethodsperformanceBenchmark,
} from '@neural-trader/predictor/testing';

// Generate test data
const { predictions, actuals } = generateSyntheticData({
    size: 10000,
    distribution: 'normal',
    noise: 0.1,
});

// Evaluate coverage
const results = evaluateCoverage(predictor, predictions, actuals);
console.log(`Coverage: ${results.empiricalCoverage}%`);
console.log(`Expected: ${(1 - predictor.alpha) * 100}%`);

// Compare methods
const comparison = await compareMethods({
    methods: ['conformal', 'bootstrap', 'mcDropout'],
    testData: { predictions, actuals },
    metrics: ['coverage', 'width', 'latency'],
});

// Benchmark
const bench = await performanceBenchmark(predictor, {
    iterations: 10000,
    calibrationSizes: [1000, 2000, 5000],
});
```

## 📚 Examples

All examples will be provided in:
- `examples/` (Rust)
- `packages/predictor/examples/` (TypeScript)

Topics covered:
- Basic conformal prediction
- Adaptive conformal inference
- Conformalized quantile regression
- Real-time trading integration
- Performance optimization
- Monitoring and alerting
- Comparison with other methods
