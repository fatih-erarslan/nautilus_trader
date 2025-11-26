# Training Guide

Best practices for training neural forecasting models with `nt-neural`.

## Table of Contents

1. [Training Configuration](#training-configuration)
2. [Data Preparation](#data-preparation)
3. [Training Process](#training-process)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Monitoring and Debugging](#monitoring-and-debugging)
6. [Advanced Techniques](#advanced-techniques)
7. [Production Deployment](#production-deployment)

## Training Configuration

### Basic Configuration

```rust
use nt_neural::{Trainer, TrainingConfig};

let config = TrainingConfig {
    // Core settings
    batch_size: 32,
    num_epochs: 100,
    learning_rate: 1e-3,

    // Regularization
    weight_decay: 1e-5,
    gradient_clip: Some(1.0),
    dropout: 0.2,

    // Early stopping
    early_stopping_patience: 10,
    early_stopping_min_delta: 1e-4,

    // Validation
    validation_split: 0.2,

    // Performance
    mixed_precision: true,
    num_workers: 4,
};

let trainer = Trainer::new(config);
```

### Learning Rate Scheduling

```rust
use nt_neural::training::{LRScheduler, SchedulerConfig};

let config = TrainingConfig {
    learning_rate: 1e-3,
    lr_scheduler: Some(LRScheduler::CosineAnnealing {
        t_max: 100,        // Total epochs
        eta_min: 1e-6,     // Minimum LR
    }),
    ..Default::default()
};

// Alternative: Step decay
let config = TrainingConfig {
    learning_rate: 1e-3,
    lr_scheduler: Some(LRScheduler::StepLR {
        step_size: 30,     // Decay every 30 epochs
        gamma: 0.1,        // Multiply by 0.1
    }),
    ..Default::default()
};

// Alternative: Reduce on plateau
let config = TrainingConfig {
    learning_rate: 1e-3,
    lr_scheduler: Some(LRScheduler::ReduceLROnPlateau {
        patience: 5,       // Wait 5 epochs
        factor: 0.5,       // Multiply by 0.5
        min_lr: 1e-6,
    }),
    ..Default::default()
};
```

## Data Preparation

### Time Series Dataset

```rust
use nt_neural::training::TimeSeriesDataset;
use nt_neural::utils::preprocessing::normalize;

async fn prepare_data() -> anyhow::Result<TimeSeriesDataset> {
    // Load raw data
    let prices = load_price_data().await?;

    // Normalize
    let (normalized, params) = normalize(&prices);

    // Create dataset
    let dataset = TimeSeriesDataset::new(
        normalized,
        input_size: 168,    // Use 168 hours (1 week)
        horizon: 24,        // Predict 24 hours ahead
        stride: 1,          // Sliding window stride
    )?;

    // Save normalization parameters for later
    save_params(&params, "norm_params.json")?;

    Ok(dataset)
}
```

### Train/Validation Split

```rust
use nt_neural::training::DataLoader;

fn create_data_loaders(
    dataset: TimeSeriesDataset,
    validation_split: f32,
) -> anyhow::Result<(DataLoader, DataLoader)> {
    let split_idx = (dataset.len() as f32 * (1.0 - validation_split)) as usize;

    // Time series split (no shuffling!)
    let (train_data, val_data) = dataset.split_at(split_idx);

    // Create loaders
    let train_loader = DataLoader::new(
        train_data,
        batch_size: 32,
        shuffle: false,  // Don't shuffle time series!
        num_workers: 4,
    )?;

    let val_loader = DataLoader::new(
        val_data,
        batch_size: 64,  // Larger for validation
        shuffle: false,
        num_workers: 2,
    )?;

    Ok((train_loader, val_loader))
}
```

### Feature Engineering Pipeline

```rust
use nt_neural::utils::{preprocessing::*, features::*};

fn feature_pipeline(prices: &[f64]) -> anyhow::Result<Vec<Vec<f64>>> {
    // 1. Outlier removal
    let clean = remove_outliers(prices, 3.0);

    // 2. Normalization
    let (normalized, _) = normalize(&clean);

    // 3. Lagged features
    let lags = create_lags(&normalized, &[1, 3, 7, 14, 21]);

    // 4. Rolling statistics
    let ma_20 = rolling_mean(&normalized, 20);
    let ma_50 = rolling_mean(&normalized, 50);
    let std_20 = rolling_std(&normalized, 20);

    // 5. Technical indicators
    let ema = ema(&normalized, 0.3);
    let roc = rate_of_change(&normalized, 1);

    // 6. Combine all features
    let mut features = Vec::new();
    for i in 0..lags.len() {
        let mut row = lags[i].clone();
        row.extend(&[ma_20[i], ma_50[i], std_20[i], ema[i], roc[i]]);
        features.push(row);
    }

    Ok(features)
}
```

## Training Process

### Basic Training Loop

```rust
#[cfg(feature = "candle")]
use nt_neural::{NHITSModel, Trainer, TrainingConfig};

#[cfg(feature = "candle")]
async fn train_model() -> anyhow::Result<()> {
    // 1. Initialize model
    let model = NHITSModel::new(config)?;

    // 2. Create trainer
    let train_config = TrainingConfig {
        batch_size: 32,
        num_epochs: 100,
        learning_rate: 1e-3,
        early_stopping_patience: 10,
        ..Default::default()
    };
    let trainer = Trainer::new(train_config);

    // 3. Prepare data
    let (train_loader, val_loader) = prepare_data().await?;

    // 4. Train
    let trained_model = trainer.train(
        &model,
        &train_loader,
        Some(&val_loader),
    ).await?;

    // 5. Save model
    trained_model.save_weights("model.safetensors")?;

    Ok(())
}
```

### With Callbacks

```rust
#[cfg(feature = "candle")]
use nt_neural::training::{Trainer, TrainingCallback};

#[cfg(feature = "candle")]
async fn train_with_callbacks() -> anyhow::Result<()> {
    let mut trainer = Trainer::new(config);

    // Add checkpoint callback
    trainer.add_callback(TrainingCallback::Checkpoint {
        save_dir: "./checkpoints".into(),
        save_best_only: true,
        monitor: "val_loss".to_string(),
    });

    // Add TensorBoard logging
    trainer.add_callback(TrainingCallback::TensorBoard {
        log_dir: "./logs".into(),
        update_freq: 100,  // Log every 100 steps
    });

    // Add learning rate monitoring
    trainer.add_callback(TrainingCallback::LRMonitor);

    // Train with callbacks
    let trained_model = trainer.train(&model, &train_loader, val_loader).await?;

    Ok(())
}
```

### Checkpointing

```rust
use nt_neural::storage::{AgentDbStorage, ModelCheckpoint};

async fn save_checkpoint(
    model: &impl NeuralModel,
    epoch: usize,
    loss: f64,
) -> anyhow::Result<()> {
    let storage = AgentDbStorage::new("./data/agentdb.db").await?;

    // Create checkpoint metadata
    let checkpoint = ModelCheckpoint {
        checkpoint_id: format!("checkpoint-epoch-{}", epoch),
        model_id: model.model_id().to_string(),
        epoch,
        loss,
        metrics: Some(/* training metrics */),
        timestamp: chrono::Utc::now(),
    };

    // Serialize model state
    let state_bytes = model.serialize()?;

    // Save to AgentDB
    let checkpoint_id = storage.save_checkpoint(
        &model.model_id(),
        checkpoint,
        &state_bytes,
    ).await?;

    tracing::info!("Checkpoint saved: {}", checkpoint_id);
    Ok(())
}
```

## Hyperparameter Tuning

### Grid Search

```rust
use nt_neural::utils::validation::grid_search;

async fn tune_hyperparameters() -> anyhow::Result<()> {
    // Define parameter grid
    let param_grid = vec![
        ("hidden_size", vec![128, 256, 512]),
        ("num_layers", vec![2, 3, 4]),
        ("dropout", vec![0.1, 0.2, 0.3]),
        ("learning_rate", vec![1e-4, 1e-3, 1e-2]),
    ];

    // Evaluation function
    let eval_fn = |params: &HashMap<String, f64>| async move {
        let config = create_config_from_params(params)?;
        let model = NHITSModel::new(config)?;

        let trainer = Trainer::new(training_config);
        let trained = trainer.train(&model, &train_loader, &val_loader).await?;

        // Return validation loss
        Ok(trained.val_loss)
    };

    // Run grid search
    let best_params = grid_search(&param_grid, eval_fn).await?;

    println!("Best parameters: {:?}", best_params);
    Ok(())
}
```

### Random Search

```rust
use rand::Rng;

async fn random_search(n_trials: usize) -> anyhow::Result<()> {
    let mut rng = rand::thread_rng();
    let mut best_loss = f64::INFINITY;
    let mut best_params = HashMap::new();

    for trial in 0..n_trials {
        // Sample parameters
        let hidden_size = rng.gen_range(128..=512);
        let num_layers = rng.gen_range(2..=4);
        let dropout = rng.gen_range(0.1..=0.3);
        let lr = 10f64.powf(rng.gen_range(-4.0..=-2.0));

        // Create config
        let config = ModelConfig {
            hidden_size,
            dropout,
            ..Default::default()
        };

        // Train and evaluate
        let model = NHITSModel::new(config)?;
        let trainer = Trainer::new(TrainingConfig {
            learning_rate: lr,
            ..Default::default()
        });

        let trained = trainer.train(&model, &train_loader, &val_loader).await?;

        // Update best
        if trained.val_loss < best_loss {
            best_loss = trained.val_loss;
            best_params = /* save params */;
        }

        println!("Trial {}: loss = {:.4}", trial, trained.val_loss);
    }

    Ok(())
}
```

### Bayesian Optimization

```rust
// Using an external library like `bayesopt`
use bayesopt::{Optimizer, Bounds};

async fn bayesian_optimization() -> anyhow::Result<()> {
    // Define bounds
    let bounds = vec![
        Bounds::new(128.0, 512.0),    // hidden_size
        Bounds::new(0.0, 0.5),         // dropout
        Bounds::new(-4.0, -2.0),       // log(learning_rate)
    ];

    // Create optimizer
    let mut optimizer = Optimizer::new(bounds);

    // Objective function
    let objective = |params: &[f64]| async move {
        let config = ModelConfig {
            hidden_size: params[0] as usize,
            dropout: params[1],
            ..Default::default()
        };
        let lr = 10f64.powf(params[2]);

        // Train and return negative loss (to maximize)
        let loss = train_and_evaluate(&config, lr).await?;
        Ok(-loss)
    };

    // Run optimization
    for _ in 0..50 {
        let (params, _score) = optimizer.next().await?;
        let score = objective(&params).await?;
        optimizer.tell(score);
    }

    let best = optimizer.best();
    println!("Best parameters: {:?}", best);

    Ok(())
}
```

## Monitoring and Debugging

### Training Metrics

```rust
use nt_neural::training::TrainingMetrics;

fn log_metrics(metrics: &TrainingMetrics, epoch: usize) {
    println!("Epoch {}/{}", epoch, metrics.total_epochs);
    println!("  Train Loss: {:.4}", metrics.train_loss);
    println!("  Val Loss:   {:.4}", metrics.val_loss);
    println!("  Train MAE:  {:.4}", metrics.train_mae);
    println!("  Val MAE:    {:.4}", metrics.val_mae);
    println!("  Learning Rate: {:.6}", metrics.learning_rate);
    println!("  Time: {:.2}s", metrics.epoch_time_secs);
}
```

### TensorBoard Integration

```rust
use tensorboard_rs::{SummaryWriter};

async fn train_with_tensorboard() -> anyhow::Result<()> {
    let mut writer = SummaryWriter::new("./logs")?;

    for epoch in 0..num_epochs {
        let metrics = train_epoch(&model, &train_loader).await?;

        // Log scalars
        writer.add_scalar("loss/train", metrics.train_loss, epoch)?;
        writer.add_scalar("loss/val", metrics.val_loss, epoch)?;
        writer.add_scalar("mae/train", metrics.train_mae, epoch)?;
        writer.add_scalar("mae/val", metrics.val_mae, epoch)?;
        writer.add_scalar("learning_rate", metrics.learning_rate, epoch)?;

        // Log histograms
        writer.add_histogram("predictions", &predictions, epoch)?;

        // Log images (for attention weights, etc.)
        // writer.add_image("attention", &attention_map, epoch)?;
    }

    writer.close()?;
    Ok(())
}
```

### Gradient Monitoring

```rust
fn check_gradients(model: &impl NeuralModel) -> anyhow::Result<()> {
    let grad_stats = model.gradient_stats()?;

    // Check for vanishing gradients
    if grad_stats.mean_abs < 1e-7 {
        tracing::warn!("Vanishing gradients detected: {:.2e}", grad_stats.mean_abs);
    }

    // Check for exploding gradients
    if grad_stats.max_abs > 10.0 {
        tracing::warn!("Exploding gradients detected: {:.2}", grad_stats.max_abs);
    }

    // Log gradient norm
    tracing::info!("Gradient norm: {:.4}", grad_stats.norm);

    Ok(())
}
```

### Debugging Checklist

**Loss not decreasing:**
1. Check learning rate (try 1e-4, 1e-3, 1e-2)
2. Verify data normalization
3. Check for data leakage
4. Simplify model (reduce hidden_size)
5. Check gradient flow

**Overfitting:**
1. Increase dropout (0.2 → 0.3 → 0.4)
2. Add weight decay (1e-5 → 1e-4)
3. Reduce model capacity
4. Add more training data
5. Use early stopping

**Training instability:**
1. Enable gradient clipping (1.0)
2. Reduce learning rate
3. Add batch normalization
4. Check for NaN in data
5. Use mixed precision carefully

## Advanced Techniques

### Transfer Learning

```rust
async fn transfer_learning() -> anyhow::Result<()> {
    // Load pretrained model
    let mut model = NHITSModel::load("pretrained_model.safetensors")?;

    // Freeze early layers
    model.freeze_layers(0..3)?;

    // Fine-tune on new data
    let config = TrainingConfig {
        learning_rate: 1e-4,  // Lower LR for fine-tuning
        num_epochs: 20,        // Fewer epochs
        ..Default::default()
    };

    let trainer = Trainer::new(config);
    let finetuned = trainer.train(&model, &new_data_loader, val_loader).await?;

    Ok(())
}
```

### Multi-Task Learning

```rust
#[cfg(feature = "candle")]
struct MultiTaskModel {
    shared_encoder: NHITSModel,
    task1_head: Dense,  // Price prediction
    task2_head: Dense,  // Volatility prediction
}

#[cfg(feature = "candle")]
impl MultiTaskModel {
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let encoded = self.shared_encoder.encode(x)?;
        let price_pred = self.task1_head.forward(&encoded)?;
        let vol_pred = self.task2_head.forward(&encoded)?;
        Ok((price_pred, vol_pred))
    }

    fn loss(&self, price_pred: &Tensor, vol_pred: &Tensor,
            price_true: &Tensor, vol_true: &Tensor) -> Result<Tensor> {
        let price_loss = mse_loss(price_pred, price_true)?;
        let vol_loss = mse_loss(vol_pred, vol_true)?;

        // Weighted combination
        let total_loss = (price_loss * 0.7 + vol_loss * 0.3)?;
        Ok(total_loss)
    }
}
```

### Ensemble Training

```rust
async fn train_ensemble(n_models: usize) -> anyhow::Result<()> {
    let mut models = Vec::new();

    for i in 0..n_models {
        // Create model with different random seed
        let model = create_model_with_seed(i)?;

        // Train with different data splits or hyperparameters
        let trained = train_model(&model).await?;
        models.push(trained);
    }

    // Ensemble prediction
    fn ensemble_predict(models: &[Model], input: &Tensor) -> Result<Tensor> {
        let predictions: Vec<Tensor> = models.iter()
            .map(|m| m.forward(input))
            .collect::<Result<Vec<_>>>()?;

        // Average predictions
        let stacked = Tensor::stack(&predictions, 0)?;
        let mean = stacked.mean(0)?;
        Ok(mean)
    }

    Ok(())
}
```

### Quantization-Aware Training

```rust
use nt_neural::training::QuantizationConfig;

async fn train_quantized() -> anyhow::Result<()> {
    let config = TrainingConfig {
        quantization: Some(QuantizationConfig {
            bits: 8,                    // 8-bit quantization
            symmetric: true,
            per_channel: true,
            calibration_samples: 1000,
        }),
        ..Default::default()
    };

    let trainer = Trainer::new(config);
    let model = trainer.train(&model, &train_loader, val_loader).await?;

    // Model is now quantized and ~4x smaller
    model.save_quantized("model_int8.safetensors")?;

    Ok(())
}
```

## Production Deployment

### Model Export

```rust
use nt_neural::storage::AgentDbStorage;

async fn export_for_production(model: &impl NeuralModel) -> anyhow::Result<()> {
    let storage = AgentDbStorage::new("./models/agentdb.db").await?;

    // Create metadata
    let metadata = ModelMetadata {
        name: "btc-predictor-v1".to_string(),
        model_type: "NHITS".to_string(),
        version: "1.0.0".to_string(),
        description: Some("Production BTC predictor".to_string()),
        tags: vec!["production".to_string(), "crypto".to_string()],
        hyperparameters: Some(serde_json::to_value(&model.config())?),
        metrics: Some(/* final metrics */),
        ..Default::default()
    };

    // Serialize model
    let model_bytes = model.to_safetensors()?;

    // Save to AgentDB
    let model_id = storage.save_model(&model_bytes, metadata).await?;

    // Also save normalization parameters
    save_norm_params("norm_params.json")?;

    // Save feature config
    save_feature_config("feature_config.json")?;

    println!("Model exported: {}", model_id);
    Ok(())
}
```

### Validation Suite

```rust
async fn production_validation(model: &impl NeuralModel) -> anyhow::Result<()> {
    // 1. Backtesting on historical data
    let backtest_results = backtest(model, &historical_data).await?;
    assert!(backtest_results.sharpe_ratio > 1.0);

    // 2. Walk-forward validation
    let wf_results = walk_forward_validation(model, &data, n_splits).await?;
    assert!(wf_results.mean_mae < threshold);

    // 3. Stress testing
    let stress_results = stress_test(model, &extreme_scenarios).await?;
    assert!(stress_results.max_loss < max_acceptable_loss);

    // 4. Latency test
    let latency = measure_inference_latency(model).await?;
    assert!(latency < Duration::from_millis(10));

    // 5. Consistency check
    let consistency = check_prediction_consistency(model).await?;
    assert!(consistency > 0.95);

    println!("All production validation checks passed!");
    Ok(())
}
```

## Next Steps

- [Inference Guide](INFERENCE.md) - Deployment and serving
- [AgentDB Guide](AGENTDB.md) - Model storage and versioning
- [API Reference](API.md) - Complete API documentation
- [Examples](../../neural-trader-rust/crates/neural/examples/) - Code examples
