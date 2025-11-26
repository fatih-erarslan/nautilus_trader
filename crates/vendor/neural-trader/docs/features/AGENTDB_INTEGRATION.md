# AgentDB Integration for Neural Trader

## Overview

This document describes the integration of AgentDB into the Neural Trader Rust project for persistent model storage, versioning, and vector-based similarity search.

## What is AgentDB?

AgentDB is a frontier memory features database with MCP integration that provides:

- **Vector Search**: 150x faster vector similarity search with HNSW indexing
- **Reflexion Memory**: Store episodes with self-critique for learning
- **Skill Library**: Reusable skills discovered from successful patterns
- **Causal Reasoning**: Track cause-effect relationships
- **QUIC Sync**: Multi-agent coordination with real-time synchronization

## Architecture

### Module Structure

```
neural-trader-rust/crates/neural/src/storage/
├── mod.rs                 # Module exports
├── types.rs              # Type definitions
└── agentdb.rs            # AgentDB integration
```

### Key Components

#### 1. Storage Types (`types.rs`)

- **ModelMetadata**: Rich metadata for model storage including:
  - Model name, type, and version
  - Training metrics (loss, accuracy, etc.)
  - Architecture information (layers, parameters)
  - Tags for categorization
  - Custom metadata fields

- **ModelCheckpoint**: Training checkpoint management:
  - Epoch and step tracking
  - Loss snapshots
  - Optimizer state preservation

- **SearchFilter**: Query models by:
  - Model type
  - Tags
  - Performance metrics
  - Creation dates

- **SimilarityMetric**: Vector search metrics:
  - Cosine similarity (default)
  - Euclidean distance
  - Dot product

#### 2. AgentDB Storage (`agentdb.rs`)

Main storage backend with async API:

```rust
pub struct AgentDbStorage {
    config: AgentDbConfig,
    initialized: bool,
}
```

### Core Features

#### Model Storage

```rust
// Save a model with metadata
let model_id = storage.save_model(
    &model_bytes,
    ModelMetadata {
        name: "nhits-btc-24h".to_string(),
        model_type: "NHITS".to_string(),
        version: "1.0.0".to_string(),
        tags: vec!["crypto".to_string(), "btc".to_string()],
        metrics: Some(TrainingMetrics {
            train_loss: 0.0234,
            val_loss: 0.0267,
            training_time: 3600.0,
            epochs: 100,
            ..Default::default()
        }),
        ..Default::default()
    }
).await?;

// Load the model
let model_bytes = storage.load_model(&model_id).await?;
```

#### Vector Similarity Search

```rust
// Generate embedding for query
let query_embedding: Vec<f32> = /* embedding model output */;

// Search for similar models
let similar = storage.search_similar_models(&query_embedding, k: 5).await?;

for result in similar {
    println!("Model: {}, Score: {:.4}",
             result.metadata.name,
             result.score);
}
```

#### Checkpoint Management

```rust
// Save training checkpoint
let checkpoint = ModelCheckpoint {
    checkpoint_id: uuid::Uuid::new_v4().to_string(),
    model_id: model_id.clone(),
    epoch: 10,
    step: 1000,
    loss: 0.123,
    val_loss: Some(0.145),
    optimizer_state: Some(optimizer_state_json),
    created_at: chrono::Utc::now(),
};

let checkpoint_id = storage
    .save_checkpoint(&model_id, checkpoint, &state_bytes)
    .await?;

// Resume training from checkpoint
let (checkpoint, state_bytes) = storage
    .load_checkpoint(&checkpoint_id)
    .await?;
```

## CLI Integration

The integration shells out to `npx agentdb` commands:

### Initialization

```bash
npx agentdb init ./data/models/agentdb.db \
    --dimension 768 \
    --preset medium
```

### Vector Search

```bash
npx agentdb vector-search ./data/models/agentdb.db \
    "[0.1,0.2,0.3,...]" \
    -k 10 \
    -m cosine \
    -f json
```

### Reflexion Storage

```bash
npx agentdb reflexion store \
    session-id \
    task-name \
    reward \
    success \
    critique-json
```

## Configuration

```rust
let config = AgentDbConfig {
    db_path: PathBuf::from("./data/models/agentdb.db"),
    dimension: 768,  // Vector dimension (default: 768)
    preset: "medium".to_string(),  // small|medium|large
    in_memory: false,  // Use :memory: for testing
};

let storage = AgentDbStorage::with_config(config).await?;
```

## Examples

### Basic Usage

```bash
cargo run --example agentdb_basic
```

Demonstrates:
- Initializing storage
- Saving and loading models
- Managing metadata
- Database statistics

### Similarity Search

```bash
cargo run --example agentdb_similarity_search
```

Demonstrates:
- Creating model collections
- Vector-based similarity search
- Filtering by tags and types
- Comparing similarity metrics

### Checkpoint Management

```bash
cargo run --example agentdb_checkpoints
```

Demonstrates:
- Saving training checkpoints
- Resuming from checkpoints
- Best checkpoint selection
- Checkpoint cleanup strategies

## Testing

### Unit Tests

```bash
cargo test --package nt-neural --lib storage
```

### Integration Tests

Requires `npx agentdb` to be installed:

```bash
cargo test --package nt-neural --test storage_integration_test -- --ignored
```

## Dependencies

Added to `Cargo.toml`:

```toml
[dependencies]
# AgentDB integration
tempfile = "3.8"      # Temporary files for model storage
fasthash = "0.4"      # Fast hashing for embeddings
```

## Coordination Hooks

The implementation integrates with Claude Flow coordination:

```bash
# Before work
npx claude-flow@alpha hooks pre-task \
    --description "AgentDB integration"

# After edits
npx claude-flow@alpha hooks post-edit \
    --file "agentdb.rs" \
    --memory-key "swarm/backend/agentdb-done"

# After completion
npx claude-flow@alpha hooks post-task \
    --task-id "agentdb"
```

## Performance Characteristics

- **Vector Search**: 150x faster with HNSW indexing
- **Storage**: Efficient binary model storage with SafeTensors
- **Metadata**: JSON-based flexible metadata storage
- **Scalability**: Supports small (<10K), medium (10K-100K), large (>100K) presets

## Future Enhancements

1. **Embedding Models**: Integrate proper embedding models (sentence-transformers, OpenAI)
2. **Batch Operations**: Bulk model save/load operations
3. **Model Compression**: Quantization integration (4-32x reduction)
4. **Distributed Sync**: QUIC-based multi-agent synchronization
5. **Causal Tracking**: Track model performance causality
6. **Skill Consolidation**: Auto-create reusable training patterns

## References

- [AgentDB Documentation](https://agentdb.ruv.io)
- [AgentDB GitHub](https://github.com/ruvnet/agentdb)
- [Neural Trader Project](https://github.com/your-org/neural-trader)

## Error Handling

The implementation provides comprehensive error handling:

```rust
pub enum NeuralError {
    StorageError(String),
    // ... other errors
}
```

All AgentDB operations return `Result<T, NeuralError>` with detailed error messages.

## Best Practices

1. **Initialize Once**: Create storage instance at application startup
2. **Batch Operations**: Use list_models() for bulk queries
3. **Checkpoint Strategy**: Keep N best + latest checkpoints
4. **Metadata Tags**: Use consistent tagging scheme
5. **Vector Dimensions**: Match your embedding model (768 for BERT, 1536 for OpenAI)
6. **Preset Selection**: Choose based on expected model count

## Troubleshooting

### AgentDB Not Found

```bash
npm install -g agentdb
# or
npx agentdb --version
```

### Permission Issues

```bash
chmod +x $(which agentdb)
```

### Database Corruption

```bash
# Export data
npx agentdb export ./agentdb.db ./backup.json --compress

# Recreate database
rm ./agentdb.db
npx agentdb init ./agentdb.db

# Import data
npx agentdb import ./backup.json.gz --decompress
```

## License

MIT License - See main project LICENSE file
