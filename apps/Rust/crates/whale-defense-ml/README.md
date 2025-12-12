# Whale Defense ML

Ultra-fast Transformer-based whale detection system for high-frequency trading with sub-500μs inference targets.

## Features

- **Transformer Architecture**: 6-layer transformer with multi-head attention for sequence modeling
- **Real-time Performance**: Optimized for <500μs inference latency
- **Ensemble Prediction**: Combines transformer with anomaly detection for robust predictions
- **Feature Engineering**: Comprehensive market feature extraction including technical indicators
- **Interpretability**: Built-in feature importance and attention visualization
- **Production Ready**: Async streaming support, performance monitoring, and metrics collection

## Architecture

### Core Components

1. **Transformer Model** (`transformer.rs`)
   - Multi-head self-attention mechanism
   - Positional encoding for temporal information
   - 6 transformer layers with 8 attention heads
   - Optimized for GPU inference with Candle framework

2. **Feature Extraction** (`features.rs`)
   - Price and volume features
   - Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
   - Market microstructure (spread, VWAP)
   - Statistical features (volatility, skewness, kurtosis)

3. **Ensemble Predictor** (`ensemble.rs`)
   - Weighted combination of models
   - Anomaly detection integration
   - Confidence scoring based on model agreement
   - Threat level assessment (1-5 scale)

4. **Performance Metrics** (`metrics.rs`)
   - Real-time inference timing
   - Accuracy, precision, recall, F1 score
   - ROC/AUC calculation
   - Performance violation tracking

## Usage

### Basic Example

```rust
use whale_defense_ml::{WhaleDetector, WhaleDetectorBuilder};
use candle_core::Device;

// Create detector with default settings
let detector = WhaleDetector::new()?;

// Or use builder for custom configuration
let detector = WhaleDetectorBuilder::new()
    .device(Device::Cuda(0))
    .sequence_length(60)
    .weights_path("models/whale_detector.pth")
    .build()?;

// Process market tick
let prediction = detector.process_tick(
    price: 45000.0,
    volume: 1_000_000.0,
    bid: Some(44999.5),
    ask: Some(45000.5),
).await?;

if let Some(result) = prediction {
    println!("Whale probability: {:.2}%", result.whale_probability * 100.0);
    println!("Threat level: {}/5", result.threat_level);
    println!("Inference time: {}μs", result.inference_time_us);
}
```

### Streaming Integration

```rust
use whale_defense_ml::{WhaleDetectorStream, MarketTick, WhaleAlert};
use tokio::sync::mpsc;

// Create channels
let (tick_tx, tick_rx) = mpsc::channel(1000);
let (alert_tx, mut alert_rx) = mpsc::channel(100);

// Create stream processor
let stream = WhaleDetectorStream::new(
    Arc::new(detector),
    tick_rx,
    alert_tx,
);

// Run detector in background
tokio::spawn(stream.run());

// Handle alerts
while let Some(alert) = alert_rx.recv().await {
    println!("🚨 Whale Alert!");
    println!("  Probability: {:.2}%", alert.whale_probability * 100.0);
    println!("  Threat Level: {}/5", alert.threat_level);
}
```

## Performance

The system is optimized for ultra-low latency inference:

- **Target**: <500μs per inference
- **Typical**: 200-400μs on modern CPUs
- **GPU**: <100μs on NVIDIA GPUs with CUDA
- **Batch**: Efficient batch processing for historical analysis

### Optimization Techniques

1. **Model Quantization**: FP16 inference where supported
2. **Kernel Fusion**: Optimized attention computation
3. **Memory Pooling**: Reused tensors for reduced allocation
4. **Parallel Feature Extraction**: SIMD-optimized indicators

## Training

Training data should include labeled whale events with market data:

```rust
use whale_defense_ml::{WhaleDataset, WhaleEvent, WhaleEventType};

// Create dataset
let dataset = WhaleDataset::new(
    &prices,
    &volumes,
    &whale_events,
    sequence_length: 60,
    prediction_horizon: 15,
)?;

// Split data
let (train, test) = dataset.train_test_split(0.2)?;

// Train model (requires additional training code)
```

## Model Architecture Details

### Transformer Configuration
- Input dimension: 19 features
- Hidden dimension: 256
- Attention heads: 8
- Encoder layers: 6
- Feed-forward multiplier: 4x
- Dropout rate: 0.1
- Max sequence length: 60

### Feature Vector (19 dimensions)
1. Price
2. Volume
3. SMA(20)
4. EMA(20)
5. RSI(14)
6. Bollinger Band Position
7. MACD
8. VWAP
9. Volume SMA(20)
10. Volume Ratio
11. Spread
12. Relative Spread
13. 1-minute price change
14. 5-minute price change
15. Volatility
16. Skewness
17. Kurtosis
18. Bid price
19. Ask price

## Production Deployment

1. **Model Loading**: Pre-trained weights can be loaded from disk
2. **Performance Monitoring**: Built-in metrics collection and alerting
3. **Error Handling**: Graceful degradation on inference failures
4. **Resource Management**: Automatic memory cleanup and pooling
5. **Scaling**: Thread-safe design for multi-threaded inference

## Testing

Run tests with:
```bash
cargo test --features test
```

Run benchmarks:
```bash
cargo bench --features benchmark
```

## License

MIT OR Apache-2.0