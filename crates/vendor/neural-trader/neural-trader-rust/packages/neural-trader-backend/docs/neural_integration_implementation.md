# Neural Module nt-neural Crate Integration

## Overview
This document provides implementation details for integrating the nt-neural crate into the NAPI backend's neural.rs module.

## Integration Architecture

### Model Registry
A global registry stores trained models in memory using:
```rust
type ModelRegistry = Arc<RwLock<HashMap<String, Arc<ModelInfo>>>>;
```

### Key Components

#### 1. Neural Forecasting (`neural_forecast`)
- Uses `Predictor<M>` from nt-neural
- Supports multiple model types: NHITS, LSTM-Attention, GRU, TCN
- Provides confidence intervals via quantile predictions
- GPU acceleration when available (CUDA/Metal)

#### 2. Neural Training (`neural_train`)
- Uses model-specific trainers (e.g., `NHITSTrainer`)
- Configures training with `TrainingConfig` and `OptimizerConfig`
- Supports early stopping and checkpointing
- Returns unique model IDs and stores in registry

#### 3. Model Evaluation (`neural_evaluate`)
- Loads models from registry
- Reads test data via Polars DataFrame
- Calculates MAE, RMSE, MAPE, R² metrics
- Supports GPU acceleration

#### 4. Model Status (`neural_model_status`)
- Queries registry for model metadata
- Returns training status, accuracy, timestamps
- Supports querying all models or specific model by ID

#### 5. Hyperparameter Optimization (`neural_optimize`)
- Grid search, random search, or Bayesian optimization
- Uses nt-neural's `Optimizer` components
- Configurable trial counts and parameter ranges

#### 6. Backtesting (`neural_backtest`)
- Historical performance evaluation
- Integrates with nt-backtesting crate
- Calculates Sharpe ratio, max drawdown, win rate

## Model Types Supported

| Model | Description | Best For |
|-------|-------------|----------|
| NHITS | Neural Hierarchical Interpolation | Multi-horizon forecasting |
| LSTM-Attention | LSTM with attention mechanism | Sequence dependencies |
| GRU | Gated Recurrent Unit | Faster than LSTM |
| TCN | Temporal Convolutional Network | Parallel processing |
| DeepAR | Probabilistic forecasting | Uncertainty quantification |
| N-BEATS | Pure MLP architecture | Interpretable forecasts |
| Transformer | Self-attention mechanism | Long-range dependencies |

## GPU Acceleration

Automatic device selection:
1. CUDA (NVIDIA) if available
2. Metal (Apple Silicon) if available
3. Accelerate (Apple CPU optimization)
4. Standard CPU fallback

## Error Handling

All neural operations use `NeuralTraderError::Neural` variant:
```rust
NeuralTraderError::Neural(format!("Error message: {}", details))
```

## Dependencies Required

### Cargo.toml additions:
```toml
nt-neural = { path = "../../crates/neural", features = ["candle"] }
lazy_static = "1.4"
uuid = { version = "1.6", features = ["v4"] }
polars = { version = "0.35", features = ["lazy", "csv-file"] }
```

## Implementation Status

- [x] Model registry architecture
- [x] Neural forecasting with confidence intervals
- [x] Model training with checkpointing
- [x] Model evaluation metrics
- [x] Model status queries
- [x] Hyperparameter optimization framework
- [x] Backtesting framework
- [ ] Model persistence to AgentDB
- [ ] Real-time streaming inference
- [ ] Model ensemble predictions

## Testing

Create test data:
```bash
cargo run --example basic_training --features candle
```

Run inference:
```bash
cargo run --example inference_example --features candle
```

## Performance Targets

- Inference latency: <10ms (single prediction)
- Training throughput: >1000 samples/sec (GPU)
- Model load time: <100ms
- Memory usage: <2GB per model

## Security Considerations

1. Validate all file paths (no path traversal)
2. Limit model file sizes (<500MB)
3. Sanitize model IDs (alphanumeric + hyphens only)
4. Rate-limit training requests
5. Sandbox model execution

## Future Enhancements

1. Model versioning and rollback
2. Distributed training across multiple GPUs
3. Automated model retraining pipelines
4. Model compression and quantization
5. Transfer learning support
