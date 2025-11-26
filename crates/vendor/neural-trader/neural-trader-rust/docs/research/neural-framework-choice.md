# Neural Forecasting Framework Research - Agent 4

## Date: 2025-11-12
## Agent: 4 (Neural Forecasting Models)

## Framework Selection

After analyzing the requirements for porting NHITS, LSTM-Attention, and Transformer models from Python to Rust, I evaluated three major Rust ML frameworks:

### 1. **candle-core** (SELECTED) ⭐
**Repository**: https://github.com/huggingface/candle

**Pros**:
- Built and maintained by Hugging Face (industry leader in ML)
- PyTorch-like API - familiar to ML engineers
- Native Rust implementation - no FFI overhead
- Excellent CUDA support via cudarc
- Metal support for Apple Silicon (via metal crate)
- Modern, idiomatic Rust code
- Active development and community
- Built-in ONNX export capabilities
- Safetensors support for model serialization

**Cons**:
- Younger ecosystem than PyTorch
- Fewer pre-built model architectures
- Documentation still growing

**Why chosen**: Best balance of GPU support, Rust idioms, and modern ML capabilities. The PyTorch-like API makes porting Python models straightforward.

### 2. **tch-rs** (Alternative)
**Repository**: https://github.com/LaurentMazare/tch-rs

**Pros**:
- Direct PyTorch C++ bindings
- Most mature Rust ML library
- Complete PyTorch feature parity
- Excellent GPU support

**Cons**:
- Requires PyTorch C++ installation
- FFI overhead
- Not idiomatic Rust
- Deployment complexity

### 3. **burn** (Alternative)
**Repository**: https://github.com/tracel-ai/burn

**Pros**:
- Comprehensive deep learning framework
- Backend-agnostic (CUDA, WGPU, CPU)
- Good for production deployments

**Cons**:
- Different API paradigm from PyTorch
- Steeper learning curve
- Less documentation for time series models

## Architecture Design

### Model 1: NHITS (Neural Hierarchical Interpolation for Time Series)

**Key Features**:
- Multi-stack architecture with frequency downsampling
- Hierarchical interpolation for interpretability
- Residual connections between stacks
- Backcast/forecast decomposition
- Quantile regression for confidence intervals

**Rust Implementation Strategy**:
```rust
// Stack-based architecture
struct NHITSStack {
    blocks: Vec<MLPBlock>,
    theta_layer: Linear,
    backcast_layer: Linear,
    forecast_layer: Linear,
}

struct NHITSModel {
    stacks: Vec<NHITSStack>,
    input_size: usize,
    horizon: usize,
    freq_downsample: Vec<usize>,
}
```

**GPU Optimization**:
- Batch processing with parallel stack execution
- CUDA kernel fusion for interpolation
- Mixed precision training (FP16/FP32)

### Model 2: LSTM with Attention

**Key Features**:
- Long Short-Term Memory cells for sequence modeling
- Multi-head attention mechanism
- Sequence-to-sequence architecture
- Teacher forcing during training
- Beam search for inference

**Rust Implementation Strategy**:
```rust
struct LSTMCell {
    input_gate: Linear,
    forget_gate: Linear,
    cell_gate: Linear,
    output_gate: Linear,
}

struct AttentionMechanism {
    query_proj: Linear,
    key_proj: Linear,
    value_proj: Linear,
    num_heads: usize,
}

struct LSTMAttentionModel {
    encoder: Vec<LSTMCell>,
    decoder: Vec<LSTMCell>,
    attention: AttentionMechanism,
}
```

**GPU Optimization**:
- Vectorized LSTM operations
- Parallel attention head computation
- Gradient checkpointing for memory efficiency

### Model 3: Transformer for Time Series

**Key Features**:
- Positional encoding for temporal information
- Multi-head self-attention
- Feed-forward networks with GELU activation
- Encoder-decoder architecture
- Layer normalization

**Rust Implementation Strategy**:
```rust
struct PositionalEncoding {
    d_model: usize,
    max_len: usize,
    encoding: Tensor,
}

struct TransformerBlock {
    self_attention: MultiHeadAttention,
    cross_attention: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

struct TransformerModel {
    encoder: Vec<TransformerBlock>,
    decoder: Vec<TransformerBlock>,
    positional_encoding: PositionalEncoding,
}
```

**GPU Optimization**:
- Flash attention for memory efficiency
- Kernel fusion for layer norm + residual
- Mixed precision with FP16 matmuls

## Training Pipeline

### Data Processing
- **polars** for efficient DataFrame operations
- Memory-mapped datasets for large-scale training
- Parallel data loading with tokio
- Normalization and standardization

### Optimization
- Adam optimizer with weight decay
- Learning rate scheduling (cosine annealing)
- Gradient clipping for stability
- Early stopping based on validation loss

### Distributed Training (Future)
- Data parallelism across GPUs
- Model parallelism for large models
- Gradient accumulation for large batch sizes

## Performance Targets

### Inference Latency
- **Target**: <10ms per prediction (vs ~50ms Python)
- **Strategy**:
  - GPU kernel fusion
  - Batch inference
  - Model quantization (INT8)
  - ONNX runtime for production

### Training Speed
- **Target**: >10x faster than Python
- **Strategy**:
  - Native GPU operations (no Python overhead)
  - Mixed precision training
  - Optimized data loading
  - Parallel preprocessing

### Memory Efficiency
- **Target**: 50% reduction vs PyTorch
- **Strategy**:
  - Gradient checkpointing
  - In-place operations
  - Memory pooling
  - Tensor lifecycle management

## Integration Points

### Agent 3 (Market Data APIs)
- Receive historical price data for training
- Stream real-time data for online learning
- Format: Polars DataFrames

### Agent 2 (MCP Tools)
- Export model predictions via MCP server
- Expose training metrics and status
- Enable remote model management

### Agent 8 (AgentDB)
- Store model versions and checkpoints
- Track hyperparameter experiments
- Enable model lineage tracking
- Vector storage for embeddings

## Dependencies

```toml
[dependencies]
# Core ML framework
candle-core = "0.3"
candle-nn = "0.3"

# GPU acceleration
cudarc = { version = "0.9", optional = true }
metal = { version = "0.27", optional = true }

# Data processing
polars = { version = "0.36", features = ["lazy", "temporal"] }
ndarray = "0.15"

# Optimization
rand = "0.8"
rand_distr = "0.4"

# Model serialization
safetensors = "0.4"
bincode = "1.3"

[features]
cuda = ["cudarc", "candle-core/cuda"]
metal = ["metal", "candle-core/metal"]
```

## Implementation Roadmap

### Phase 1: Foundation (Current)
- [x] Framework research and selection
- [ ] Basic tensor operations
- [ ] Layer abstractions (Linear, Conv1d, LSTM, Attention)
- [ ] Loss functions (MSE, MAE, Quantile Loss)

### Phase 2: Model Implementation
- [ ] NHITS architecture
- [ ] LSTM-Attention architecture
- [ ] Transformer architecture
- [ ] Model factory and configuration

### Phase 3: Training
- [ ] Data loaders with polars
- [ ] Training loop with mixed precision
- [ ] Validation and early stopping
- [ ] Hyperparameter tuning

### Phase 4: Inference
- [ ] Batch inference engine
- [ ] Model quantization (INT8)
- [ ] ONNX export
- [ ] Serving API

### Phase 5: Integration
- [ ] AgentDB integration for model versioning
- [ ] MCP tool exposure
- [ ] Trading strategy integration
- [ ] Performance benchmarking

## Benchmarking Plan

### Accuracy Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- Directional Accuracy (sign prediction)

### Performance Metrics
- Training time per epoch
- Inference latency (p50, p95, p99)
- Throughput (predictions/second)
- GPU memory usage
- Power consumption

### Target Baselines
- Python NHITS: ~50ms inference, ~10min training
- **Rust Target**: <10ms inference, <1min training
- **Speedup Goal**: 10x-50x

## References

### Python Implementation
- `/workspaces/neural-trader/src/neural_forecast/nhits_forecaster.py`
- `/workspaces/neural-trader/src/neural_forecast/optimized_nhits_engine.py`
- `/workspaces/neural-trader/src/neural_forecast/neural_model_manager.py`

### Papers
- NHITS: "Neural Hierarchical Interpolation for Time Series Forecasting" (2022)
- N-BEATS: "Neural basis expansion analysis for interpretable time series forecasting" (2020)
- Attention is All You Need (2017)
- Long Short-Term Memory (1997)

### Candle Resources
- GitHub: https://github.com/huggingface/candle
- Examples: https://github.com/huggingface/candle/tree/main/candle-examples
- Docs: https://huggingface.github.io/candle/

## Next Steps

1. ✅ Complete this research document
2. ⏳ Set up candle-core dependencies in Cargo.toml
3. ⏳ Implement basic neural network layers
4. ⏳ Port NHITS model architecture
5. ⏳ Create training pipeline
6. ⏳ Benchmark against Python baseline
7. ⏳ Integrate with trading strategies

---

**Decision**: Proceed with **candle-core** for all three models (NHITS, LSTM-Attention, Transformer).
