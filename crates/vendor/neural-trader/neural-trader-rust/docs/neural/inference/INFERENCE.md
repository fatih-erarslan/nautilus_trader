# Inference Guide

Production-ready inference with `nt-neural` for low-latency predictions.

## Table of Contents

1. [Overview](#overview)
2. [Single Predictions](#single-predictions)
3. [Batch Predictions](#batch-predictions)
4. [Streaming Predictions](#streaming-predictions)
5. [Optimization](#optimization)
6. [Production Deployment](#production-deployment)
7. [Monitoring](#monitoring)

## Overview

The `nt-neural` crate provides three inference modes:

- **Single**: One prediction at a time
- **Batch**: Multiple predictions efficiently
- **Streaming**: Real-time continuous predictions

### Performance Targets

| Model | Single Latency | Batch Throughput | Memory |
|-------|---------------|------------------|--------|
| NHITS | <5ms | >1000/sec | ~200MB |
| TCN | <3ms | >2000/sec | ~150MB |
| LSTM-Attention | <15ms | >500/sec | ~400MB |
| Transformer | <10ms | >800/sec | ~600MB |

## Single Predictions

### Basic Usage

```rust
#[cfg(feature = "candle")]
use nt_neural::{Predictor, PredictionResult};
use nt_neural::storage::AgentDbStorage;

#[cfg(feature = "candle")]
async fn single_prediction() -> anyhow::Result<()> {
    // Load model from AgentDB
    let storage = AgentDbStorage::new("./models/agentdb.db").await?;
    let model_bytes = storage.load_model("model-id").await?;

    // Deserialize model
    let model = NHITSModel::from_safetensors(&model_bytes)?;

    // Create predictor
    let predictor = Predictor::new(model)?;

    // Prepare input (168 hours of data)
    let input_data = vec![/* 168 data points */];

    // Make prediction
    let result: PredictionResult = predictor.predict(&input_data).await?;

    println!("Prediction: {:?}", result.values);
    println!("Confidence intervals: {:?}", result.intervals);
    println!("Inference time: {}ms", result.inference_time_ms);

    Ok(())
}
```

### With Preprocessing

```rust
use nt_neural::utils::preprocessing::*;
use nt_neural::utils::features::*;

async fn predict_with_preprocessing(
    raw_prices: &[f64],
) -> anyhow::Result<Vec<f64>> {
    // 1. Load normalization parameters (saved during training)
    let norm_params = load_norm_params("norm_params.json")?;

    // 2. Normalize input
    let normalized = normalize_with_params(raw_prices, &norm_params);

    // 3. Create features
    let features = create_prediction_features(&normalized)?;

    // 4. Run inference
    let predictor = load_predictor("model-id").await?;
    let result = predictor.predict(&features).await?;

    // 5. Denormalize predictions
    let predictions = denormalize(&result.values, &norm_params);

    Ok(predictions)
}
```

### Error Handling

```rust
use nt_neural::{Predictor, NeuralError};

async fn robust_prediction(input: &[f64]) -> anyhow::Result<Vec<f64>> {
    let predictor = load_predictor("model-id").await?;

    match predictor.predict(input).await {
        Ok(result) => {
            // Validate predictions
            if result.values.iter().any(|&x| x.is_nan()) {
                return Err(anyhow::anyhow!("NaN in predictions"));
            }

            // Check confidence
            if result.confidence.unwrap_or(1.0) < 0.7 {
                tracing::warn!("Low confidence prediction: {}", result.confidence.unwrap());
            }

            Ok(result.values)
        }
        Err(NeuralError::InputShapeMismatch { expected, got }) => {
            Err(anyhow::anyhow!("Wrong input size: expected {}, got {}", expected, got))
        }
        Err(e) => Err(e.into()),
    }
}
```

## Batch Predictions

### Efficient Batching

```rust
#[cfg(feature = "candle")]
use nt_neural::BatchPredictor;

#[cfg(feature = "candle")]
async fn batch_predictions() -> anyhow::Result<()> {
    // Create batch predictor
    let batch_predictor = BatchPredictor::new(
        model,
        batch_size: 32,
    )?;

    // Prepare multiple inputs
    let inputs: Vec<Vec<f64>> = vec![
        // Input 1
        vec![/* 168 data points */],
        // Input 2
        vec![/* 168 data points */],
        // ... more inputs
    ];

    // Batch predict (automatically batches and manages memory)
    let results = batch_predictor.predict_batch(&inputs).await?;

    for (i, result) in results.iter().enumerate() {
        println!("Prediction {}: {:?}", i, result.values);
    }

    // Throughput: ~1000 predictions/sec on GPU
    println!("Throughput: {:.2} pred/sec", batch_predictor.throughput());

    Ok(())
}
```

### Parallel Batch Processing

```rust
use rayon::prelude::*;

async fn parallel_batch_predictions(
    inputs: Vec<Vec<f64>>,
) -> anyhow::Result<Vec<PredictionResult>> {
    // Split into chunks for parallel processing
    let chunk_size = 100;
    let chunks: Vec<_> = inputs.chunks(chunk_size).collect();

    // Process chunks in parallel
    let results: Vec<_> = chunks
        .par_iter()
        .map(|chunk| {
            let predictor = load_predictor("model-id")?;
            predictor.predict_batch(chunk)
        })
        .collect::<Result<Vec<_>>>()?;

    // Flatten results
    Ok(results.into_iter().flatten().collect())
}
```

## Streaming Predictions

### Real-Time Stream

```rust
#[cfg(feature = "candle")]
use nt_neural::inference::StreamingPredictor;
use tokio::sync::mpsc;

#[cfg(feature = "candle")]
async fn streaming_predictions() -> anyhow::Result<()> {
    // Create streaming predictor
    let mut predictor = StreamingPredictor::new(
        model,
        window_size: 168,
        horizon: 24,
    )?;

    // Channel for incoming data
    let (tx, mut rx) = mpsc::channel::<f64>(1000);

    // Spawn prediction task
    tokio::spawn(async move {
        while let Some(new_value) = rx.recv().await {
            // Update internal buffer
            predictor.update(new_value);

            // Make prediction when buffer is full
            if predictor.is_ready() {
                match predictor.predict().await {
                    Ok(result) => {
                        println!("Streaming prediction: {:?}", result.values);
                    }
                    Err(e) => {
                        tracing::error!("Prediction error: {}", e);
                    }
                }
            }
        }
    });

    // Simulate incoming data stream
    for price in price_stream() {
        tx.send(price).await?;
    }

    Ok(())
}
```

### WebSocket Integration

```rust
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::StreamExt;

async fn websocket_predictions() -> anyhow::Result<()> {
    // Connect to data source
    let (ws_stream, _) = connect_async("wss://api.example.com/prices").await?;
    let (_, mut read) = ws_stream.split();

    // Create streaming predictor
    let mut predictor = StreamingPredictor::new(model, 168, 24)?;

    // Process WebSocket messages
    while let Some(msg) = read.next().await {
        match msg? {
            Message::Text(text) => {
                // Parse price from message
                let price: f64 = serde_json::from_str(&text)?;

                // Update predictor
                predictor.update(price);

                // Predict when ready
                if predictor.is_ready() {
                    let prediction = predictor.predict().await?;

                    // Send prediction somewhere (websocket, queue, etc.)
                    send_prediction(&prediction).await?;
                }
            }
            _ => {}
        }
    }

    Ok(())
}
```

## Optimization

### Model Quantization

```rust
use nt_neural::optimization::quantize_model;

async fn optimize_model() -> anyhow::Result<()> {
    // Load full precision model
    let model = load_model("model-fp32.safetensors")?;

    // Quantize to INT8 (~4x smaller, ~2x faster)
    let quantized = quantize_model(
        &model,
        bits: 8,
        calibration_data: &calibration_samples,
    )?;

    // Save quantized model
    quantized.save("model-int8.safetensors")?;

    // Compare accuracy
    let fp32_results = benchmark(&model, &test_data).await?;
    let int8_results = benchmark(&quantized, &test_data).await?;

    println!("FP32 MAE: {:.4}", fp32_results.mae);
    println!("INT8 MAE: {:.4}", int8_results.mae);
    println!("Accuracy loss: {:.2}%",
        (int8_results.mae - fp32_results.mae) / fp32_results.mae * 100.0);

    Ok(())
}
```

### Model Pruning

```rust
use nt_neural::optimization::prune_model;

async fn prune_for_inference() -> anyhow::Result<()> {
    let model = load_model("model.safetensors")?;

    // Prune 30% of weights
    let pruned = prune_model(
        &model,
        sparsity: 0.3,
        method: PruningMethod::Magnitude,
    )?;

    // Fine-tune pruned model
    let finetuned = finetune_pruned(&pruned, &train_data).await?;

    // ~30% smaller, similar accuracy
    finetuned.save("model-pruned.safetensors")?;

    Ok(())
}
```

### Batch Size Tuning

```rust
async fn find_optimal_batch_size() -> anyhow::Result<usize> {
    let model = load_model("model.safetensors")?;
    let test_data = load_test_data()?;

    let mut best_batch_size = 1;
    let mut best_throughput = 0.0;

    // Test different batch sizes
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128] {
        let predictor = BatchPredictor::new(&model, batch_size)?;

        let start = std::time::Instant::now();
        let _ = predictor.predict_batch(&test_data).await?;
        let elapsed = start.elapsed().as_secs_f64();

        let throughput = test_data.len() as f64 / elapsed;

        println!("Batch size {}: {:.2} pred/sec", batch_size, throughput);

        if throughput > best_throughput {
            best_throughput = throughput;
            best_batch_size = batch_size;
        }
    }

    println!("Optimal batch size: {}", best_batch_size);
    Ok(best_batch_size)
}
```

### Caching

```rust
use lru::LruCache;
use std::sync::Arc;
use tokio::sync::Mutex;

struct CachedPredictor {
    predictor: Arc<Predictor>,
    cache: Arc<Mutex<LruCache<Vec<u8>, PredictionResult>>>,
}

impl CachedPredictor {
    fn new(predictor: Predictor, cache_size: usize) -> Self {
        Self {
            predictor: Arc::new(predictor),
            cache: Arc::new(Mutex::new(LruCache::new(cache_size))),
        }
    }

    async fn predict(&self, input: &[f64]) -> anyhow::Result<PredictionResult> {
        // Create cache key (hash of input)
        let key = hash_input(input);

        // Check cache
        {
            let mut cache = self.cache.lock().await;
            if let Some(cached) = cache.get(&key) {
                return Ok(cached.clone());
            }
        }

        // Cache miss - compute prediction
        let result = self.predictor.predict(input).await?;

        // Store in cache
        {
            let mut cache = self.cache.lock().await;
            cache.put(key, result.clone());
        }

        Ok(result)
    }
}
```

## Production Deployment

### REST API Server

```rust
use axum::{
    extract::{State, Json},
    routing::post,
    Router,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct PredictionRequest {
    input: Vec<f64>,
}

#[derive(Serialize)]
struct PredictionResponse {
    prediction: Vec<f64>,
    confidence: f64,
    inference_time_ms: f64,
}

struct AppState {
    predictor: Arc<Predictor>,
}

async fn predict_endpoint(
    State(state): State<Arc<AppState>>,
    Json(request): Json<PredictionRequest>,
) -> Result<Json<PredictionResponse>, StatusCode> {
    // Validate input
    if request.input.len() != 168 {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Make prediction
    let result = state.predictor
        .predict(&request.input)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(PredictionResponse {
        prediction: result.values,
        confidence: result.confidence.unwrap_or(1.0),
        inference_time_ms: result.inference_time_ms,
    }))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load model
    let model = load_model("model.safetensors").await?;
    let predictor = Predictor::new(model)?;

    // Create app state
    let state = Arc::new(AppState {
        predictor: Arc::new(predictor),
    });

    // Build router
    let app = Router::new()
        .route("/predict", post(predict_endpoint))
        .with_state(state);

    // Start server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    axum::serve(listener, app).await?;

    Ok(())
}
```

### gRPC Service

```rust
use tonic::{transport::Server, Request, Response, Status};

pub mod prediction {
    tonic::include_proto!("prediction");
}

use prediction::{
    prediction_server::{Prediction, PredictionServer},
    PredictRequest, PredictResponse,
};

pub struct PredictionService {
    predictor: Arc<Predictor>,
}

#[tonic::async_trait]
impl Prediction for PredictionService {
    async fn predict(
        &self,
        request: Request<PredictRequest>,
    ) -> Result<Response<PredictResponse>, Status> {
        let input = request.into_inner().input;

        let result = self.predictor
            .predict(&input)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(PredictResponse {
            prediction: result.values,
            confidence: result.confidence.unwrap_or(1.0),
        }))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = load_model("model.safetensors").await?;
    let predictor = Predictor::new(model)?;

    let service = PredictionService {
        predictor: Arc::new(predictor),
    };

    Server::builder()
        .add_service(PredictionServer::new(service))
        .serve("0.0.0.0:50051".parse()?)
        .await?;

    Ok(())
}
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM rust:1.75 as builder

WORKDIR /app
COPY . .

# Build with optimizations
RUN cargo build --release --package nt-neural --features candle,cuda

FROM nvidia/cuda:12.1-runtime-ubuntu22.04

WORKDIR /app

# Copy binary and model
COPY --from=builder /app/target/release/predictor-server .
COPY models/ ./models/

# Expose port
EXPOSE 3000

# Run server
CMD ["./predictor-server"]
```

```bash
# Build and run
docker build -t predictor-server .
docker run -p 3000:3000 --gpus all predictor-server
```

## Monitoring

### Metrics Collection

```rust
use prometheus::{Counter, Histogram, Registry};

struct InferenceMetrics {
    prediction_count: Counter,
    prediction_latency: Histogram,
    prediction_errors: Counter,
}

impl InferenceMetrics {
    fn new(registry: &Registry) -> Self {
        let prediction_count = Counter::new(
            "predictions_total",
            "Total number of predictions",
        ).unwrap();
        registry.register(Box::new(prediction_count.clone())).unwrap();

        let prediction_latency = Histogram::with_opts(
            histogram_opts!(
                "prediction_latency_seconds",
                "Prediction latency in seconds",
                vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
            )
        ).unwrap();
        registry.register(Box::new(prediction_latency.clone())).unwrap();

        let prediction_errors = Counter::new(
            "prediction_errors_total",
            "Total prediction errors",
        ).unwrap();
        registry.register(Box::new(prediction_errors.clone())).unwrap();

        Self {
            prediction_count,
            prediction_latency,
            prediction_errors,
        }
    }

    async fn predict_with_metrics(
        &self,
        predictor: &Predictor,
        input: &[f64],
    ) -> anyhow::Result<PredictionResult> {
        let timer = self.prediction_latency.start_timer();

        let result = match predictor.predict(input).await {
            Ok(r) => {
                self.prediction_count.inc();
                Ok(r)
            }
            Err(e) => {
                self.prediction_errors.inc();
                Err(e.into())
            }
        };

        timer.observe_duration();
        result
    }
}
```

### Health Checks

```rust
use axum::http::StatusCode;

async fn health_check(
    State(state): State<Arc<AppState>>,
) -> StatusCode {
    // Check if model is loaded
    if !state.predictor.is_loaded() {
        return StatusCode::SERVICE_UNAVAILABLE;
    }

    // Check if GPU is available (if needed)
    #[cfg(feature = "cuda")]
    if !state.predictor.device().is_cuda() {
        return StatusCode::SERVICE_UNAVAILABLE;
    }

    // Quick inference test
    let test_input = vec![0.0; 168];
    match state.predictor.predict(&test_input).await {
        Ok(_) => StatusCode::OK,
        Err(_) => StatusCode::SERVICE_UNAVAILABLE,
    }
}
```

### Alerting

```rust
use tracing::{error, warn};

async fn monitor_predictions(predictor: &Predictor) {
    let mut error_count = 0;
    let mut slow_count = 0;

    loop {
        tokio::time::sleep(Duration::from_secs(60)).await;

        let stats = predictor.get_stats().await;

        // Check error rate
        if stats.error_rate > 0.05 {
            error!("High error rate: {:.2}%", stats.error_rate * 100.0);
            error_count += 1;

            if error_count >= 3 {
                // Trigger alert
                send_alert("High prediction error rate").await;
            }
        } else {
            error_count = 0;
        }

        // Check latency
        if stats.p99_latency_ms > 50.0 {
            warn!("High P99 latency: {:.2}ms", stats.p99_latency_ms);
            slow_count += 1;

            if slow_count >= 3 {
                send_alert("High prediction latency").await;
            }
        } else {
            slow_count = 0;
        }
    }
}
```

## Next Steps

- [AgentDB Integration](AGENTDB.md) - Model storage and versioning
- [API Reference](API.md) - Complete API documentation
- [Training Guide](TRAINING.md) - Model training best practices
- [Examples](../../neural-trader-rust/crates/neural/examples/) - Code examples
