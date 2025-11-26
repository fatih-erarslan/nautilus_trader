# AgentDB Integration Guide

Complete guide to using AgentDB for neural model storage, versioning, and similarity search.

## Table of Contents

1. [Overview](#overview)
2. [Setup](#setup)
3. [Model Storage](#model-storage)
4. [Model Retrieval](#model-retrieval)
5. [Similarity Search](#similarity-search)
6. [Checkpointing](#checkpointing)
7. [Version Control](#version-control)
8. [Best Practices](#best-practices)

## Overview

AgentDB provides:

- **Vector Storage**: Store models with semantic embeddings
- **Similarity Search**: Find similar models by architecture/performance
- **Checkpointing**: Save training states for resumption
- **Versioning**: Track model evolution over time
- **Metadata**: Rich searchable metadata

### Architecture

```
┌─────────────────┐
│  Neural Model   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│   AgentDB API   │─────▶│  SQLite DB   │
└─────────────────┘      └──────────────┘
         │
         ▼
┌─────────────────┐
│  Vector Index   │ (HNSW for fast search)
└─────────────────┘
```

## Setup

### Initialize AgentDB

```rust
use nt_neural::storage::AgentDbStorage;

async fn setup_storage() -> anyhow::Result<()> {
    // Basic initialization
    let storage = AgentDbStorage::new("./data/models/agentdb.db").await?;

    // With custom configuration
    let config = AgentDbConfig {
        db_path: PathBuf::from("./models/agentdb.db"),
        dimension: 768,           // Embedding dimension
        preset: "large".to_string(),
        in_memory: false,         // Use disk storage
    };

    let storage = AgentDbStorage::with_config(config).await?;

    Ok(())
}
```

### Configuration Options

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `db_path` | `./data/models/agentdb.db` | Any path | Database file location |
| `dimension` | 768 | 128-2048 | Vector embedding size |
| `preset` | `medium` | `small`, `medium`, `large` | Size optimization |
| `in_memory` | false | true/false | RAM vs disk storage |

## Model Storage

### Save Model

```rust
use nt_neural::storage::{AgentDbStorage, ModelMetadata};

async fn save_trained_model(
    model: &impl NeuralModel,
    training_metrics: &TrainingMetrics,
) -> anyhow::Result<String> {
    let storage = AgentDbStorage::new("./models/agentdb.db").await?;

    // Create comprehensive metadata
    let metadata = ModelMetadata {
        name: "btc-hourly-nhits-v1".to_string(),
        model_type: "NHITS".to_string(),
        version: "1.0.0".to_string(),

        description: Some(
            "Bitcoin hourly price predictor using NHITS architecture. \
             Trained on 2 years of data with 168h input, 24h horizon."
                .to_string()
        ),

        tags: vec![
            "production".to_string(),
            "crypto".to_string(),
            "hourly".to_string(),
            "btc".to_string(),
        ],

        hyperparameters: Some(serde_json::json!({
            "input_size": 168,
            "horizon": 24,
            "hidden_size": 512,
            "num_stacks": 3,
            "learning_rate": 1e-3,
            "batch_size": 32,
        })),

        training_config: Some(serde_json::json!({
            "num_epochs": 100,
            "early_stopping": true,
            "optimizer": "Adam",
        })),

        metrics: Some(serde_json::json!({
            "train_loss": training_metrics.train_loss,
            "val_loss": training_metrics.val_loss,
            "mae": training_metrics.mae,
            "rmse": training_metrics.rmse,
            "r2_score": training_metrics.r2_score,
        })),

        dataset_info: Some(serde_json::json!({
            "source": "Binance",
            "timeframe": "1h",
            "train_samples": 17520,
            "val_samples": 4380,
            "date_range": "2022-01-01 to 2024-01-01",
        })),

        hardware: Some(serde_json::json!({
            "gpu": "NVIDIA RTX 4090",
            "memory": "24GB",
            "training_time_hours": 2.5,
        })),

        ..Default::default()
    };

    // Serialize model to SafeTensors format
    let model_bytes = model.to_safetensors()?;

    // Save to AgentDB
    let model_id = storage.save_model(&model_bytes, metadata).await?;

    tracing::info!("Model saved with ID: {}", model_id);
    Ok(model_id)
}
```

### Metadata Structure

```rust
pub struct ModelMetadata {
    // Required fields
    pub name: String,
    pub model_type: String,
    pub version: String,

    // Optional but recommended
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub hyperparameters: Option<serde_json::Value>,
    pub training_config: Option<serde_json::Value>,
    pub metrics: Option<serde_json::Value>,
    pub dataset_info: Option<serde_json::Value>,
    pub hardware: Option<serde_json::Value>,

    // System fields (auto-populated)
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub model_size_bytes: usize,
}
```

## Model Retrieval

### Load by ID

```rust
use nt_neural::storage::AgentDbStorage;

async fn load_model_by_id(model_id: &str) -> anyhow::Result<NHITSModel> {
    let storage = AgentDbStorage::new("./models/agentdb.db").await?;

    // Get metadata
    let metadata = storage.get_metadata(model_id).await?;
    println!("Loading model: {}", metadata.name);
    println!("Type: {}", metadata.model_type);
    println!("Version: {}", metadata.version);

    // Load model bytes
    let model_bytes = storage.load_model(model_id).await?;

    // Deserialize
    let model = NHITSModel::from_safetensors(&model_bytes)?;

    Ok(model)
}
```

### List All Models

```rust
use nt_neural::storage::{AgentDbStorage, SearchFilter};

async fn list_all_models() -> anyhow::Result<()> {
    let storage = AgentDbStorage::new("./models/agentdb.db").await?;

    // Get all models
    let models = storage.list_models(None).await?;

    println!("Found {} models:", models.len());
    for metadata in models {
        println!("\n- {}", metadata.name);
        println!("  ID: {}", metadata.model_id);
        println!("  Type: {}", metadata.model_type);
        println!("  Version: {}", metadata.version);

        if let Some(metrics) = &metadata.metrics {
            if let Some(val_loss) = metrics.get("val_loss") {
                println!("  Val Loss: {:.4}", val_loss);
            }
        }
    }

    Ok(())
}
```

### Filter Models

```rust
use nt_neural::storage::SearchFilter;

async fn find_models_with_filter() -> anyhow::Result<()> {
    let storage = AgentDbStorage::new("./models/agentdb.db").await?;

    // Create filter
    let filter = SearchFilter {
        model_type: Some("NHITS".to_string()),
        tags: Some(vec!["production".to_string(), "crypto".to_string()]),
        min_val_loss: Some(0.01),
        max_val_loss: Some(0.05),
        created_after: Some(chrono::Utc::now() - chrono::Duration::days(30)),
        ..Default::default()
    };

    // Search with filter
    let models = storage.list_models(Some(filter)).await?;

    println!("Found {} matching models", models.len());
    for metadata in models {
        println!("- {}: {}", metadata.name, metadata.val_loss.unwrap_or(0.0));
    }

    Ok(())
}
```

## Similarity Search

### Search by Embedding

```rust
use nt_neural::storage::{AgentDbStorage, SimilarityMetric};

async fn find_similar_models() -> anyhow::Result<()> {
    let storage = AgentDbStorage::new("./models/agentdb.db").await?;

    // Generate embedding for query (from model architecture, hyperparams, etc.)
    let query_embedding = generate_model_embedding(&query_config)?;

    // Search for top 5 similar models
    let similar = storage.search_similar_models(&query_embedding, 5).await?;

    println!("Similar models:");
    for result in similar {
        println!("\n{} (similarity: {:.4})", result.metadata.name, result.score);
        println!("  Type: {}", result.metadata.model_type);
        println!("  Hyperparameters: {:?}", result.metadata.hyperparameters);
    }

    Ok(())
}
```

### Custom Similarity Metrics

```rust
use nt_neural::storage::SimilarityMetric;

async fn search_with_custom_metric() -> anyhow::Result<()> {
    let storage = AgentDbStorage::new("./models/agentdb.db").await?;

    let embedding = generate_embedding()?;

    // Cosine similarity (default, best for normalized vectors)
    let cosine_results = storage.search_similar_models_with_metric(
        &embedding,
        k: 10,
        metric: SimilarityMetric::Cosine,
    ).await?;

    // Euclidean distance (L2 norm)
    let l2_results = storage.search_similar_models_with_metric(
        &embedding,
        k: 10,
        metric: SimilarityMetric::L2,
    ).await?;

    // Manhattan distance (L1 norm)
    let l1_results = storage.search_similar_models_with_metric(
        &embedding,
        k: 10,
        metric: SimilarityMetric::L1,
    ).await?;

    Ok(())
}
```

### Find Best Performing Models

```rust
async fn find_best_models(n: usize) -> anyhow::Result<Vec<ModelMetadata>> {
    let storage = AgentDbStorage::new("./models/agentdb.db").await?;

    // Get all models
    let mut models = storage.list_models(None).await?;

    // Sort by validation loss
    models.sort_by(|a, b| {
        let a_loss = a.val_loss.unwrap_or(f64::INFINITY);
        let b_loss = b.val_loss.unwrap_or(f64::INFINITY);
        a_loss.partial_cmp(&b_loss).unwrap()
    });

    // Take top N
    Ok(models.into_iter().take(n).collect())
}
```

## Checkpointing

### Save Training Checkpoint

```rust
use nt_neural::storage::{AgentDbStorage, ModelCheckpoint};

async fn save_training_checkpoint(
    model_id: &str,
    epoch: usize,
    loss: f64,
    optimizer_state: &OptimizerState,
) -> anyhow::Result<String> {
    let storage = AgentDbStorage::new("./models/agentdb.db").await?;

    // Create checkpoint metadata
    let checkpoint = ModelCheckpoint {
        checkpoint_id: format!("ckpt-{}-epoch-{}", model_id, epoch),
        model_id: model_id.to_string(),
        epoch,
        loss,
        metrics: Some(serde_json::json!({
            "train_loss": 0.045,
            "val_loss": loss,
            "learning_rate": 1e-3,
        })),
        optimizer_config: Some(serde_json::json!({
            "optimizer": "Adam",
            "beta1": 0.9,
            "beta2": 0.999,
        })),
        timestamp: chrono::Utc::now(),
    };

    // Serialize checkpoint state (model + optimizer)
    let state = CheckpointState {
        model_weights: model.state_dict()?,
        optimizer_state: optimizer_state.clone(),
    };
    let state_bytes = bincode::serialize(&state)?;

    // Save checkpoint
    let checkpoint_id = storage.save_checkpoint(
        model_id,
        checkpoint,
        &state_bytes,
    ).await?;

    tracing::info!("Checkpoint saved: {}", checkpoint_id);
    Ok(checkpoint_id)
}
```

### Resume from Checkpoint

```rust
async fn resume_training_from_checkpoint(
    checkpoint_id: &str,
) -> anyhow::Result<(NHITSModel, OptimizerState, usize)> {
    let storage = AgentDbStorage::new("./models/agentdb.db").await?;

    // Load checkpoint
    let (checkpoint, state_bytes) = storage.load_checkpoint(checkpoint_id).await?;

    println!("Resuming from epoch {}", checkpoint.epoch);
    println!("Previous loss: {:.4}", checkpoint.loss);

    // Deserialize state
    let state: CheckpointState = bincode::deserialize(&state_bytes)?;

    // Restore model
    let mut model = NHITSModel::new(config)?;
    model.load_state_dict(&state.model_weights)?;

    // Restore optimizer
    let optimizer_state = state.optimizer_state;

    Ok((model, optimizer_state, checkpoint.epoch))
}
```

### Automatic Checkpointing

```rust
use nt_neural::training::{Trainer, TrainingCallback};

async fn train_with_auto_checkpoint() -> anyhow::Result<()> {
    let mut trainer = Trainer::new(config);

    // Add checkpoint callback
    trainer.add_callback(TrainingCallback::AgentDbCheckpoint {
        storage: AgentDbStorage::new("./models/agentdb.db").await?,
        save_frequency: 5,         // Save every 5 epochs
        save_best_only: true,      // Only save when val_loss improves
        keep_last_n: 3,            // Keep last 3 checkpoints
    });

    // Train (checkpoints saved automatically)
    let trained = trainer.train(&model, &train_loader, val_loader).await?;

    Ok(())
}
```

## Version Control

### Semantic Versioning

```rust
async fn create_new_model_version(
    base_model_id: &str,
    changes: &str,
) -> anyhow::Result<String> {
    let storage = AgentDbStorage::new("./models/agentdb.db").await?;

    // Load base model metadata
    let base_metadata = storage.get_metadata(base_model_id).await?;

    // Parse current version
    let version = Version::parse(&base_metadata.version)?;

    // Increment version based on changes
    let new_version = if changes.contains("breaking") {
        version.increment_major()
    } else if changes.contains("feature") {
        version.increment_minor()
    } else {
        version.increment_patch()
    };

    // Create new version metadata
    let new_metadata = ModelMetadata {
        name: format!("{}-{}", base_metadata.name, new_version),
        version: new_version.to_string(),
        description: Some(format!(
            "Version {} - Changes: {}",
            new_version, changes
        )),
        parent_model_id: Some(base_model_id.to_string()),
        ..base_metadata
    };

    // Save new version
    let model_bytes = load_improved_model()?;
    let new_id = storage.save_model(&model_bytes, new_metadata).await?;

    Ok(new_id)
}
```

### Model Lineage

```rust
async fn get_model_lineage(model_id: &str) -> anyhow::Result<Vec<ModelMetadata>> {
    let storage = AgentDbStorage::new("./models/agentdb.db").await?;

    let mut lineage = Vec::new();
    let mut current_id = model_id.to_string();

    // Walk back through parent models
    loop {
        let metadata = storage.get_metadata(&current_id).await?;
        lineage.push(metadata.clone());

        match metadata.parent_model_id {
            Some(parent_id) => current_id = parent_id,
            None => break,
        }
    }

    // Reverse to get chronological order
    lineage.reverse();

    // Print lineage
    println!("Model lineage:");
    for (i, meta) in lineage.iter().enumerate() {
        println!("{}: {} (v{})", i, meta.name, meta.version);
    }

    Ok(lineage)
}
```

## Best Practices

### 1. Comprehensive Metadata

Always include rich metadata for better searchability:

```rust
let metadata = ModelMetadata {
    name: "descriptive-name-with-context".to_string(),

    // Detailed description
    description: Some(format!(
        "Model for {}. Trained on {} samples from {} to {}. \
         Key features: {}. Known limitations: {}.",
        use_case, n_samples, start_date, end_date, features, limitations
    )),

    // Extensive tags
    tags: vec![
        "production".to_string(),      // Deployment status
        "crypto".to_string(),           // Domain
        "btc-usd".to_string(),         // Asset
        "hourly".to_string(),          // Timeframe
        "v1".to_string(),              // Version family
        "high-accuracy".to_string(),   // Performance tier
    ],

    // Complete hyperparameters
    hyperparameters: Some(serde_json::json!({
        // All model config
    })),

    // Full metrics
    metrics: Some(serde_json::json!({
        "mae": 0.0234,
        "rmse": 0.0456,
        "mape": 2.34,
        "r2_score": 0.92,
        "directional_accuracy": 0.58,
    })),

    ..Default::default()
};
```

### 2. Consistent Naming Convention

```rust
// Format: {asset}-{timeframe}-{model}-v{version}-{date}
let name = format!(
    "{}-{}-{}-v{}-{}",
    "btc-usd",
    "1h",
    "nhits",
    "1.0.0",
    chrono::Utc::now().format("%Y%m%d")
);
```

### 3. Regular Cleanup

```rust
async fn cleanup_old_checkpoints(keep_days: i64) -> anyhow::Result<()> {
    let storage = AgentDbStorage::new("./models/agentdb.db").await?;
    let cutoff = chrono::Utc::now() - chrono::Duration::days(keep_days);

    let models = storage.list_models(None).await?;

    for metadata in models {
        if metadata.created_at < cutoff && !metadata.tags.contains(&"production".to_string()) {
            storage.delete_model(&metadata.model_id).await?;
            tracing::info!("Deleted old model: {}", metadata.name);
        }
    }

    Ok(())
}
```

### 4. Backup and Export

```rust
async fn backup_models() -> anyhow::Result<()> {
    let storage = AgentDbStorage::new("./models/agentdb.db").await?;

    // Export database
    let backup_path = format!(
        "backups/agentdb-{}.db.gz",
        chrono::Utc::now().format("%Y%m%d")
    );

    storage.export(&backup_path, compress: true).await?;

    tracing::info!("Database backed up to {}", backup_path);
    Ok(())
}
```

### 5. Model Registry Pattern

```rust
struct ModelRegistry {
    storage: AgentDbStorage,
    cache: LruCache<String, Arc<dyn NeuralModel>>,
}

impl ModelRegistry {
    async fn get_model(&mut self, model_id: &str) -> anyhow::Result<Arc<dyn NeuralModel>> {
        // Check cache
        if let Some(model) = self.cache.get(model_id) {
            return Ok(Arc::clone(model));
        }

        // Load from storage
        let model_bytes = self.storage.load_model(model_id).await?;
        let metadata = self.storage.get_metadata(model_id).await?;

        // Deserialize based on type
        let model: Arc<dyn NeuralModel> = match metadata.model_type.as_str() {
            "NHITS" => Arc::new(NHITSModel::from_safetensors(&model_bytes)?),
            "LSTM" => Arc::new(LSTMAttentionModel::from_safetensors(&model_bytes)?),
            _ => anyhow::bail!("Unknown model type"),
        };

        // Cache and return
        self.cache.put(model_id.to_string(), Arc::clone(&model));
        Ok(model)
    }

    async fn get_best_model(&self, filter: SearchFilter) -> anyhow::Result<Arc<dyn NeuralModel>> {
        let models = self.storage.list_models(Some(filter)).await?;
        let best = models.first().ok_or_else(|| anyhow::anyhow!("No models found"))?;
        self.get_model(&best.model_id).await
    }
}
```

## Next Steps

- [API Reference](API.md) - Complete API documentation
- [Training Guide](TRAINING.md) - Training best practices
- [Inference Guide](INFERENCE.md) - Production deployment
- [Examples](../../neural-trader-rust/crates/neural/examples/) - Code examples
