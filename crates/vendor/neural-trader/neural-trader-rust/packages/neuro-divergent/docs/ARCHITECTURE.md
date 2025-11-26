# Neuro-Divergent Architecture

Comprehensive architectural documentation for the Rust-powered neural forecasting library.

## Table of Contents

- [System Overview](#system-overview)
- [Component Architecture](#component-architecture)
- [Data Flow](#data-flow)
- [Model Architecture](#model-architecture)
- [Performance Optimizations](#performance-optimizations)
- [Extension Points](#extension-points)

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    JavaScript/TypeScript API                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ NeuralFore-  │  │   Models     │  │   Utilities  │      │
│  │   caster     │  │ (LSTM, GRU,  │  │ (Metrics,    │      │
│  │              │  │  Transformer)│  │  Validation) │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          │         N-API Bindings (napi-rs)    │
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼─────────────┐
│                      Rust Core Library                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Training   │  │  Inference   │  │   Storage    │      │
│  │   Engine     │  │   Engine     │  │   (AgentDB)  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
│  ┌──────▼──────────────────▼──────────────────▼───────┐     │
│  │         Model Implementations (nt-neural)          │     │
│  │  LSTM │ GRU │ Transformer │ N-BEATS │ TCN │ ...   │     │
│  └──────┬──────────────────────────────────────────────┘    │
│         │                                                    │
│  ┌──────▼──────────────────────────────────────────────┐    │
│  │         Preprocessing & Feature Engineering         │    │
│  │  Normalization │ SIMD Ops │ Feature Gen │ Memory   │    │
│  └──────┬──────────────────────────────────────────────┘    │
└─────────┼──────────────────────────────────────────────────┘
          │
┌─────────▼──────────────────────────────────────────────────┐
│              ML Framework (Candle)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Tensor     │  │  Autograd    │  │   Backend    │     │
│  │   Ops        │  │   Engine     │  │ (CPU/CUDA/   │     │
│  │              │  │              │  │   Metal)     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **API** | JavaScript/TypeScript | User-facing interface |
| **Bindings** | napi-rs | Native Node.js bindings |
| **Core** | Rust | High-performance implementation |
| **ML Framework** | Candle | Tensor operations & autograd |
| **Acceleration** | SIMD, Rayon | Vectorization & parallelism |
| **Storage** | AgentDB | Model persistence |

---

## Component Architecture

### 1. JavaScript API Layer

#### NeuralForecaster

Main orchestration class that manages the lifecycle of forecasting operations.

```typescript
class NeuralForecaster {
    // Configuration
    private config: ForecastConfig;
    private models: Model[];
    private backend: Backend;

    // Training
    async fit(data: TimeSeriesData, options: TrainingOptions): Promise<void>;

    // Inference
    async predict(options: PredictOptions): Promise<Forecasts>;

    // Validation
    async crossValidation(options: CVOptions): Promise<CVResults>;

    // Persistence
    async saveCheckpoint(path: string): Promise<void>;
    async loadCheckpoint(path: string): Promise<void>;
}
```

**Responsibilities**:
- Data validation and preprocessing
- Model lifecycle management
- Training orchestration
- Batch prediction
- Checkpoint management

#### Model Classes

Each neural architecture has a dedicated class:

```typescript
// Recurrent models
class LSTM extends RecurrentModel { }
class GRU extends RecurrentModel { }

// Attention models
class Transformer extends AttentionModel { }
class Informer extends AttentionModel { }

// Specialized
class NBEATS extends DecomposableModel { }
class TCN extends ConvolutionalModel { }
```

### 2. N-API Bindings

Rust-to-JavaScript bridge using napi-rs:

```rust
#[napi]
pub struct NeuralForecaster {
    inner: Arc<Mutex<Forecaster>>,
}

#[napi]
impl NeuralForecaster {
    #[napi(constructor)]
    pub fn new(config: JsObject) -> Result<Self> {
        // Convert JS config to Rust
        let config = parse_config(config)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(Forecaster::new(config)?))
        })
    }

    #[napi]
    pub async fn fit(
        &self,
        data: JsObject,
        options: Option<JsObject>,
    ) -> Result<()> {
        let forecaster = self.inner.lock().unwrap();
        let data = parse_data(data)?;
        let options = parse_options(options)?;

        // Run training in Tokio runtime
        tokio::task::spawn_blocking(move || {
            forecaster.fit(&data, &options)
        }).await?
    }

    #[napi]
    pub async fn predict(&self, options: JsObject) -> Result<JsObject> {
        let forecaster = self.inner.lock().unwrap();
        let options = parse_options(options)?;

        let forecasts = forecaster.predict(&options)?;
        to_js_object(forecasts)
    }
}
```

**Key Features**:
- Zero-copy data transfer where possible
- Async/await support with Tokio
- Automatic memory management
- Type-safe conversions

### 3. Rust Core

#### Training Engine

```rust
pub struct Trainer {
    model: Box<dyn Model>,
    optimizer: Optimizer,
    loss_fn: LossFunction,
    config: TrainingConfig,
}

impl Trainer {
    pub fn train(&mut self, data: &TrainingData) -> Result<TrainingMetrics> {
        // Initialize
        let mut metrics = TrainingMetrics::new();
        let mut best_loss = f64::INFINITY;
        let mut patience_counter = 0;

        // Training loop
        for epoch in 0..self.config.epochs {
            // Forward pass (parallel batches with Rayon)
            let epoch_loss = self.train_epoch(data)?;

            // Validation
            let val_loss = self.validate(data.validation)?;

            // Early stopping
            if val_loss < best_loss {
                best_loss = val_loss;
                patience_counter = 0;
                self.save_best_model()?;
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.patience {
                    break; // Early stop
                }
            }

            metrics.push(epoch, epoch_loss, val_loss);
        }

        Ok(metrics)
    }

    fn train_epoch(&mut self, data: &TrainingData) -> Result<f64> {
        // Parallel batch processing with Rayon
        let losses: Vec<f64> = data.batches()
            .par_iter()
            .map(|batch| {
                // Forward pass
                let predictions = self.model.forward(batch.inputs)?;

                // Compute loss
                let loss = self.loss_fn.compute(&predictions, &batch.targets)?;

                // Backward pass
                self.model.backward(&loss)?;

                Ok(loss.item())
            })
            .collect::<Result<Vec<_>>>()?;

        // Aggregate and apply gradients
        let total_loss = losses.iter().sum::<f64>() / losses.len() as f64;
        self.optimizer.step()?;

        Ok(total_loss)
    }
}
```

#### Inference Engine

```rust
pub struct Predictor {
    model: Box<dyn Model>,
    config: InferenceConfig,
    memory_pool: MemoryPool,
}

impl Predictor {
    pub fn predict_batch(&self, inputs: &[Input]) -> Result<Vec<Output>> {
        // Allocate from memory pool
        let mut outputs = self.memory_pool.acquire(inputs.len());

        // Batch inference (parallelized)
        inputs.par_iter()
            .zip(outputs.par_iter_mut())
            .try_for_each(|(input, output)| {
                *output = self.predict_single(input)?;
                Ok::<(), Error>(())
            })?;

        Ok(outputs)
    }

    #[inline]
    fn predict_single(&self, input: &Input) -> Result<Output> {
        // Preprocess
        let preprocessed = self.preprocess_simd(input)?;

        // Forward pass (no gradient tracking)
        let prediction = self.model.forward_no_grad(&preprocessed)?;

        // Postprocess
        let output = self.postprocess_simd(&prediction)?;

        Ok(output)
    }
}
```

### 4. Model Implementations

#### LSTM Architecture

```rust
pub struct LSTM {
    // Layers
    input_layer: Linear,
    lstm_layers: Vec<LSTMCell>,
    dropout: Dropout,
    output_layer: Linear,

    // Configuration
    hidden_size: usize,
    num_layers: usize,
    bidirectional: bool,

    // State
    hidden_states: Option<Tensor>,
    cell_states: Option<Tensor>,
}

impl Model for LSTM {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let batch_size = input.size(0);
        let seq_len = input.size(1);

        // Initialize hidden states
        let (mut h, mut c) = self.init_hidden(batch_size)?;

        // Process sequence
        let mut outputs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let x_t = input.index_select(1, &[t])?;

            // LSTM cells
            for (layer_idx, lstm_cell) in self.lstm_layers.iter().enumerate() {
                (h[layer_idx], c[layer_idx]) = lstm_cell.forward(
                    &x_t,
                    &h[layer_idx],
                    &c[layer_idx]
                )?;
            }

            outputs.push(h.last().unwrap().clone());
        }

        // Stack outputs
        let output = Tensor::stack(&outputs, 1)?;

        // Output layer
        self.output_layer.forward(&output)
    }
}
```

#### Transformer Architecture

```rust
pub struct Transformer {
    // Encoder
    encoder_embedding: Embedding,
    encoder_layers: Vec<EncoderLayer>,
    encoder_norm: LayerNorm,

    // Decoder
    decoder_embedding: Embedding,
    decoder_layers: Vec<DecoderLayer>,
    decoder_norm: LayerNorm,

    // Output
    output_projection: Linear,

    // Configuration
    num_heads: usize,
    hidden_size: usize,
    ffn_size: usize,
}

impl Model for Transformer {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        // Encoder
        let mut enc_out = self.encoder_embedding.forward(input)?;

        for layer in &self.encoder_layers {
            enc_out = layer.forward(&enc_out)?;
        }
        enc_out = self.encoder_norm.forward(&enc_out)?;

        // Decoder (autoregressive)
        let mut dec_out = self.decoder_embedding.forward(&self.get_decoder_input())?;

        for layer in &self.decoder_layers {
            dec_out = layer.forward(&dec_out, &enc_out)?;
        }
        dec_out = self.decoder_norm.forward(&dec_out)?;

        // Project to output dimension
        self.output_projection.forward(&dec_out)
    }
}
```

### 5. SIMD Preprocessing

```rust
#[cfg(feature = "simd")]
use std::simd::{f64x4, f64x8};

pub struct Preprocessor {
    normalization: NormalizationType,
    feature_config: FeatureConfig,
}

impl Preprocessor {
    #[inline]
    pub fn normalize_simd(&self, data: &[f64]) -> Vec<f64> {
        match self.normalization {
            NormalizationType::Standard => {
                let mean = simd_mean(data);
                let std = simd_std(data, mean);
                simd_normalize(data, mean, std)
            }
            NormalizationType::MinMax => {
                let (min, max) = simd_min_max(data);
                simd_min_max_normalize(data, min, max)
            }
            NormalizationType::Robust => {
                let median = simd_median(data);
                let iqr = simd_iqr(data);
                simd_robust_scale(data, median, iqr)
            }
        }
    }

    pub fn generate_features_simd(&self, data: &[f64]) -> FeatureMatrix {
        let mut features = Vec::new();

        // SIMD rolling statistics
        if self.feature_config.rolling_mean {
            features.push(simd_rolling_mean(data, self.feature_config.window));
        }

        if self.feature_config.rolling_std {
            features.push(simd_rolling_std(data, self.feature_config.window));
        }

        if self.feature_config.ema {
            features.push(simd_ema(data, self.feature_config.alpha));
        }

        // Lag features
        if self.feature_config.lags {
            for lag in &self.feature_config.lag_values {
                features.push(create_lag(data, *lag));
            }
        }

        FeatureMatrix::from_features(features)
    }
}
```

---

## Data Flow

### Training Pipeline

```
Input Data (JSON)
        │
        ▼
┌──────────────────┐
│  Data Validation │  ← Check types, ranges, missing values
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Preprocessing   │  ← SIMD normalization, feature generation
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Batch Creation  │  ← Create training/validation batches
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Model Training  │  ← Parallel forward/backward passes
│   (Rayon)        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Checkpoint Save │  ← SafeTensors binary format
└──────────────────┘
```

### Inference Pipeline

```
Input Sequence
        │
        ▼
┌──────────────────┐
│  Preprocessing   │  ← SIMD operations, memory pool
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Model Forward   │  ← No gradient tracking, optimized path
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Postprocessing  │  ← Denormalization, SIMD
└────────┬─────────┘
         │
         ▼
    Forecasts (JSON)
```

---

## Performance Optimizations

### Memory Management

```rust
pub struct MemoryPool {
    pools: HashMap<usize, Vec<Vec<f64>>>,
    size_classes: [usize; 8],
}

impl MemoryPool {
    pub fn acquire(&mut self, size: usize) -> Vec<f64> {
        let class = self.find_size_class(size);
        self.pools.get_mut(&class)
            .and_then(|pool| pool.pop())
            .unwrap_or_else(|| Vec::with_capacity(class))
    }

    pub fn release(&mut self, mut vec: Vec<f64>) {
        vec.clear();
        let class = self.find_size_class(vec.capacity());
        self.pools.entry(class)
            .or_insert_with(Vec::new)
            .push(vec);
    }
}
```

### Parallel Processing

```rust
use rayon::prelude::*;

// Parallel batch processing
let results: Vec<_> = batches
    .par_iter()
    .map(|batch| process_batch(batch))
    .collect();

// Parallel feature generation
let features: Vec<_> = (0..num_features)
    .into_par_iter()
    .map(|i| generate_feature(data, i))
    .collect();
```

### SIMD Vectorization

```rust
#[inline]
fn simd_dot_product(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = f64x4::splat(0.0);
    let chunks_a = a.chunks_exact(4);
    let chunks_b = b.chunks_exact(4);

    for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
        let vec_a = f64x4::from_slice(chunk_a);
        let vec_b = f64x4::from_slice(chunk_b);
        sum += vec_a * vec_b;
    }

    sum.reduce_sum() +
        chunks_a.remainder().iter()
            .zip(chunks_b.remainder())
            .map(|(x, y)| x * y)
            .sum::<f64>()
}
```

---

## Extension Points

### Custom Models

Implement the `Model` trait:

```rust
pub trait Model: Send + Sync {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor>;
    fn backward(&mut self, loss: &Tensor) -> Result<()>;
    fn parameters(&self) -> Vec<&Tensor>;
    fn save(&self, path: &Path) -> Result<()>;
    fn load(&mut self, path: &Path) -> Result<()>;
}

// Example custom model
pub struct CustomModel {
    layers: Vec<Box<dyn Layer>>,
}

impl Model for CustomModel {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        for layer in &mut self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }
    // ... implement other methods
}
```

### Custom Loss Functions

```rust
pub trait LossFunction {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor>;
}

pub struct CustomLoss {
    weight: f64,
}

impl LossFunction for CustomLoss {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // Custom loss implementation
        let diff = (predictions - targets)?;
        let loss = (diff.sqr()? * self.weight)?;
        loss.mean(0)
    }
}
```

### Custom Optimizers

```rust
pub trait Optimizer {
    fn step(&mut self) -> Result<()>;
    fn zero_grad(&mut self);
}

pub struct CustomOptimizer {
    learning_rate: f64,
    parameters: Vec<Tensor>,
}

impl Optimizer for CustomOptimizer {
    fn step(&mut self) -> Result<()> {
        for param in &mut self.parameters {
            let grad = param.grad()?;
            let update = (grad * self.learning_rate)?;
            *param = (param.as_ref() - update)?;
        }
        Ok(())
    }

    fn zero_grad(&mut self) {
        for param in &mut self.parameters {
            param.zero_grad();
        }
    }
}
```

---

## Deployment Architecture

### Production Setup

```
┌──────────────────────────────────────────────────┐
│              Load Balancer                       │
└─────────────┬────────────────────────────────────┘
              │
    ┌─────────┴─────────┬──────────────┬──────────┐
    │                   │              │          │
    ▼                   ▼              ▼          ▼
┌─────────┐       ┌─────────┐    ┌─────────┐  ┌─────────┐
│ Node 1  │       │ Node 2  │    │ Node 3  │  │ Node 4  │
│ (CPU)   │       │ (CPU)   │    │ (GPU)   │  │ (GPU)   │
└─────────┘       └─────────┘    └─────────┘  └─────────┘
     │                 │              │            │
     └─────────┬───────┴──────────────┴────────────┘
               │
         ┌─────▼─────┐
         │  AgentDB  │  ← Shared model storage
         └───────────┘
```

### Scaling Considerations

- **Horizontal**: Multiple Node.js instances with load balancing
- **Vertical**: GPU acceleration for large models
- **Caching**: Pre-computed features and predictions
- **Batch Optimization**: Group predictions for efficiency

---

This architecture provides the foundation for high-performance, scalable neural forecasting with Rust and JavaScript.
