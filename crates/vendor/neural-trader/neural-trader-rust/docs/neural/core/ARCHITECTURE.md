# Neural Crate Architecture Design

## Executive Summary

This document defines the comprehensive architecture for the `nt-neural` crate, providing a unified framework for neural network models in financial time series forecasting. The architecture supports multiple backends (CPU-only, Candle, future alternatives), AgentDB integration for model storage, and distributed training capabilities.

**Version:** 1.0.0
**Status:** Design Phase
**Last Updated:** 2025-11-13

---

## 1. Current Architecture Analysis

### 1.1 Existing Structure

```
neural-trader-rust/crates/neural/
├── src/
│   ├── lib.rs              # Public API, feature gates
│   ├── error.rs            # Error handling
│   ├── stubs.rs            # Non-candle fallbacks
│   ├── models/
│   │   ├── mod.rs          # Model traits & base config
│   │   ├── nhits.rs        # NHITS implementation
│   │   ├── lstm_attention.rs # LSTM-Attention
│   │   ├── transformer.rs   # Transformer model
│   │   └── layers.rs       # Reusable layer components
│   ├── training/
│   │   ├── mod.rs          # Training config & metrics
│   │   ├── trainer.rs      # Training loop infrastructure
│   │   ├── nhits_trainer.rs # NHITS-specific training
│   │   ├── optimizer.rs    # Optimizers & schedulers
│   │   └── data_loader.rs  # Dataset & batching
│   ├── inference/
│   │   ├── mod.rs          # Inference exports
│   │   ├── predictor.rs    # Single model prediction
│   │   ├── batch.rs        # Batch inference
│   │   └── streaming.rs    # Streaming inference
│   └── utils/
│       ├── mod.rs
│       ├── preprocessing.rs # Data normalization
│       ├── metrics.rs      # Evaluation metrics
│       ├── validation.rs   # Input validation
│       └── features.rs     # Feature engineering
└── Cargo.toml
```

### 1.2 Feature Gates

**Current:**
- `default = []` - No features by default
- `candle = ["candle-core", "candle-nn"]` - Enable Candle ML framework
- `cuda = ["candle", "cudarc"]` - CUDA GPU acceleration
- `metal = ["candle"]` - Metal GPU (Apple Silicon)
- `accelerate = ["candle"]` - Accelerate framework (macOS)

**Evaluation:**
✅ Good separation of GPU backends
✅ Optional candle dependency
⚠️ Missing: Alternative backend support (Burn, SmartCore)
⚠️ Missing: Distributed training feature flag
⚠️ Missing: AgentDB integration flag

### 1.3 Identified Gaps

1. **No Unified Model Abstraction**: Each model implements its own interface
2. **Limited Backend Flexibility**: Tightly coupled to Candle
3. **No Model Registry**: No centralized model discovery/loading
4. **AgentDB Integration Missing**: No storage/versioning system
5. **Incomplete Stub Pattern**: Stubs only provide Device/Tensor, not full API
6. **No Distributed Support**: No infrastructure for multi-node training
7. **Checkpoint Strategy Unclear**: Ad-hoc checkpoint management

---

## 2. Core Design Principles

### 2.1 Architecture Principles

1. **Backend Agnostic**: Models should work with any backend (Candle, Burn, CPU-only)
2. **Graceful Degradation**: Full functionality without optional dependencies
3. **Type Safety**: Leverage Rust's type system for correctness
4. **Zero-Cost Abstractions**: No runtime overhead from abstractions
5. **Async-First**: All I/O operations are async
6. **Distributed-Ready**: Design for horizontal scaling from day one

### 2.2 Module Organization

```
nt-neural/
├── core/           # Backend-agnostic abstractions
├── backends/       # Backend implementations
├── models/         # Model architectures
├── training/       # Training infrastructure
├── inference/      # Prediction engines
├── storage/        # Model persistence (AgentDB)
├── distributed/    # Multi-node training
└── utils/          # Helper functions
```

---

## 3. Unified Model Abstraction

### 3.1 Core Trait Hierarchy

```rust
// nt-neural/src/core/model.rs

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::path::Path;

/// Core model trait - backend agnostic
#[async_trait]
pub trait Model: Send + Sync {
    /// Get model metadata
    fn metadata(&self) -> &ModelMetadata;

    /// Get model type identifier
    fn model_type(&self) -> ModelType;

    /// Total trainable parameters
    fn num_parameters(&self) -> usize;

    /// Model size in bytes (approximate)
    fn memory_footprint(&self) -> usize;

    /// Model input specification
    fn input_spec(&self) -> InputSpec;

    /// Model output specification
    fn output_spec(&self) -> OutputSpec;

    /// Validate input shape/format
    fn validate_input(&self, input: &InputData) -> Result<(), ModelError>;

    /// Save model to path (backend-specific format)
    async fn save(&self, path: impl AsRef<Path> + Send) -> Result<(), ModelError>;

    /// Load model from path
    async fn load(path: impl AsRef<Path> + Send) -> Result<Self, ModelError>
    where
        Self: Sized;

    /// Clone as Any for downcasting
    fn as_any(&self) -> &dyn Any;

    /// Clone as mutable Any
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Trainable model trait
#[async_trait]
pub trait TrainableModel: Model {
    /// Training configuration type
    type Config: TrainingConfig;

    /// Forward pass (training mode)
    fn forward_train(
        &mut self,
        input: &InputData,
        target: &TargetData,
    ) -> Result<TrainingOutput, ModelError>;

    /// Backward pass and parameter update
    fn backward(
        &mut self,
        loss: &LossValue,
        optimizer: &mut dyn Optimizer,
    ) -> Result<(), ModelError>;

    /// Set training mode
    fn set_training_mode(&mut self, training: bool);

    /// Get current learning rate
    fn learning_rate(&self) -> f64;

    /// Get trainable parameters (backend-specific)
    fn parameters_mut(&mut self) -> Box<dyn Iterator<Item = &mut dyn Parameter> + '_>;
}

/// Inference-capable model
#[async_trait]
pub trait InferenceModel: Model {
    /// Forward pass (inference mode)
    fn forward(&self, input: &InputData) -> Result<PredictionOutput, ModelError>;

    /// Batch forward pass
    fn forward_batch(&self, inputs: &[InputData]) -> Result<Vec<PredictionOutput>, ModelError> {
        inputs.iter().map(|input| self.forward(input)).collect()
    }

    /// Predict with uncertainty quantification
    fn predict_with_uncertainty(
        &self,
        input: &InputData,
        quantiles: &[f64],
    ) -> Result<UncertaintyOutput, ModelError>;
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub id: String,
    pub name: String,
    pub version: String,
    pub model_type: ModelType,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub framework: Framework,
    pub backend: Backend,
    pub tags: Vec<String>,
    pub description: Option<String>,
}

/// Model type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum ModelType {
    NHITS,
    LSTMAttention,
    Transformer,
    Custom(u64), // Hash of custom type name
}

/// Supported backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Backend {
    CpuNative,      // Pure Rust, no external deps
    Candle,         // Candle ML framework
    CandleCuda,     // Candle with CUDA
    CandleMetal,    // Candle with Metal
    Burn,           // Burn ML framework (future)
    SmartCore,      // SmartCore (future)
}

/// ML framework
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Framework {
    Native,         // Hand-written Rust
    Candle,
    Burn,
    SmartCore,
}

/// Input specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSpec {
    pub shape: Vec<Option<usize>>, // None = dynamic dimension
    pub dtype: DataType,
    pub features: Vec<FeatureSpec>,
}

/// Output specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSpec {
    pub shape: Vec<Option<usize>>,
    pub dtype: DataType,
    pub interpretation: OutputInterpretation,
}

/// Data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    F32,
    F64,
    I32,
    I64,
}

/// Feature specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSpec {
    pub name: String,
    pub dtype: DataType,
    pub normalization: Option<NormalizationType>,
}

/// Output interpretation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputInterpretation {
    PointForecast,
    QuantileForecast(Vec<f64>),
    Distribution,
    Classification(Vec<String>),
}

/// Normalization types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NormalizationType {
    ZScore,
    MinMax,
    RobustScaler,
    LogTransform,
}
```

### 3.2 Backend Abstraction Layer

```rust
// nt-neural/src/core/backend.rs

/// Backend trait for different ML frameworks
pub trait BackendOps: Send + Sync {
    /// Create a tensor from raw data
    fn create_tensor(&self, data: &[f64], shape: &[usize]) -> Result<TensorHandle, BackendError>;

    /// Matrix multiplication
    fn matmul(&self, a: &TensorHandle, b: &TensorHandle) -> Result<TensorHandle, BackendError>;

    /// Element-wise operations
    fn add(&self, a: &TensorHandle, b: &TensorHandle) -> Result<TensorHandle, BackendError>;
    fn mul(&self, a: &TensorHandle, b: &TensorHandle) -> Result<TensorHandle, BackendError>;
    fn div(&self, a: &TensorHandle, b: &TensorHandle) -> Result<TensorHandle, BackendError>;

    /// Activation functions
    fn relu(&self, x: &TensorHandle) -> Result<TensorHandle, BackendError>;
    fn gelu(&self, x: &TensorHandle) -> Result<TensorHandle, BackendError>;
    fn softmax(&self, x: &TensorHandle, dim: i64) -> Result<TensorHandle, BackendError>;

    /// Reduction operations
    fn sum(&self, x: &TensorHandle, dims: &[i64]) -> Result<TensorHandle, BackendError>;
    fn mean(&self, x: &TensorHandle, dims: &[i64]) -> Result<TensorHandle, BackendError>;

    /// Shape operations
    fn reshape(&self, x: &TensorHandle, shape: &[usize]) -> Result<TensorHandle, BackendError>;
    fn transpose(&self, x: &TensorHandle, dim0: i64, dim1: i64) -> Result<TensorHandle, BackendError>;

    /// Gradient operations (for training)
    fn backward(&self, loss: &TensorHandle) -> Result<(), BackendError>;
    fn get_gradient(&self, tensor: &TensorHandle) -> Result<TensorHandle, BackendError>;

    /// Device management
    fn device(&self) -> BackendDevice;
    fn to_device(&self, tensor: &TensorHandle, device: BackendDevice) -> Result<TensorHandle, BackendError>;
}

/// Backend-agnostic tensor handle
pub struct TensorHandle {
    inner: Box<dyn Any + Send + Sync>,
    shape: Vec<usize>,
    dtype: DataType,
    backend: Backend,
}

/// Backend device representation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendDevice {
    Cpu,
    Cuda(usize),  // GPU index
    Metal(usize),
}

/// Backend factory
pub struct BackendFactory;

impl BackendFactory {
    /// Create backend with auto-detection
    pub fn create_auto() -> Result<Box<dyn BackendOps>, BackendError> {
        #[cfg(feature = "cuda")]
        if let Ok(backend) = Self::create_cuda(0) {
            return Ok(backend);
        }

        #[cfg(feature = "metal")]
        if let Ok(backend) = Self::create_metal(0) {
            return Ok(backend);
        }

        Self::create_cpu()
    }

    /// Create CPU backend
    pub fn create_cpu() -> Result<Box<dyn BackendOps>, BackendError> {
        #[cfg(feature = "candle")]
        return Ok(Box::new(CandleCpuBackend::new()?));

        #[cfg(not(feature = "candle"))]
        return Ok(Box::new(NativeCpuBackend::new()));
    }

    /// Create CUDA backend
    #[cfg(feature = "cuda")]
    pub fn create_cuda(device: usize) -> Result<Box<dyn BackendOps>, BackendError> {
        Ok(Box::new(CandleCudaBackend::new(device)?))
    }

    /// Create Metal backend
    #[cfg(feature = "metal")]
    pub fn create_metal(device: usize) -> Result<Box<dyn BackendOps>, BackendError> {
        Ok(Box::new(CandleMetalBackend::new(device)?))
    }
}
```

### 3.3 Model Builder Pattern

```rust
// nt-neural/src/core/builder.rs

/// Model builder for consistent construction
pub struct ModelBuilder {
    model_type: ModelType,
    config: Box<dyn Any + Send + Sync>,
    backend: Option<Box<dyn BackendOps>>,
    metadata: ModelMetadata,
}

impl ModelBuilder {
    pub fn new(model_type: ModelType) -> Self {
        Self {
            model_type,
            config: Box::new(()),
            backend: None,
            metadata: ModelMetadata::default_for_type(model_type),
        }
    }

    pub fn with_config<C: 'static + Send + Sync>(mut self, config: C) -> Self {
        self.config = Box::new(config);
        self
    }

    pub fn with_backend(mut self, backend: Box<dyn BackendOps>) -> Self {
        self.backend = Some(backend);
        self
    }

    pub fn with_metadata(mut self, metadata: ModelMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn build(self) -> Result<Box<dyn Model>, ModelError> {
        let backend = self.backend.unwrap_or_else(|| {
            BackendFactory::create_auto().expect("Failed to create backend")
        });

        match self.model_type {
            ModelType::NHITS => {
                let config = self.config
                    .downcast::<NHITSConfig>()
                    .map_err(|_| ModelError::InvalidConfig)?;
                Ok(Box::new(NHITSModel::new(*config, backend, self.metadata)?))
            }
            ModelType::LSTMAttention => {
                let config = self.config
                    .downcast::<LSTMAttentionConfig>()
                    .map_err(|_| ModelError::InvalidConfig)?;
                Ok(Box::new(LSTMAttentionModel::new(*config, backend, self.metadata)?))
            }
            ModelType::Transformer => {
                let config = self.config
                    .downcast::<TransformerConfig>()
                    .map_err(|_| ModelError::InvalidConfig)?;
                Ok(Box::new(TransformerModel::new(*config, backend, self.metadata)?))
            }
            ModelType::Custom(_) => Err(ModelError::UnsupportedModelType),
        }
    }
}
```

---

## 4. AgentDB Integration Design

### 4.1 Model Storage Schema

```rust
// nt-neural/src/storage/agentdb.rs

use agentdb::{Database, Collection, Document};

/// AgentDB-backed model registry
pub struct ModelRegistry {
    db: Database,
    models_collection: Collection,
    checkpoints_collection: Collection,
    metadata_collection: Collection,
}

impl ModelRegistry {
    pub async fn new(db_path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let db = Database::open(db_path).await?;

        let models_collection = db.collection("neural_models").await?;
        let checkpoints_collection = db.collection("model_checkpoints").await?;
        let metadata_collection = db.collection("model_metadata").await?;

        Ok(Self {
            db,
            models_collection,
            checkpoints_collection,
            metadata_collection,
        })
    }

    /// Register a new model
    pub async fn register_model(
        &self,
        model: &dyn Model,
        tags: Vec<String>,
    ) -> Result<ModelId, StorageError> {
        let metadata = model.metadata();
        let doc = Document::new()
            .set("id", &metadata.id)
            .set("name", &metadata.name)
            .set("version", &metadata.version)
            .set("model_type", metadata.model_type.to_string())
            .set("created_at", metadata.created_at.to_rfc3339())
            .set("framework", format!("{:?}", metadata.framework))
            .set("backend", format!("{:?}", metadata.backend))
            .set("tags", serde_json::to_value(tags)?);

        self.models_collection.insert(doc).await?;
        Ok(ModelId(metadata.id.clone()))
    }

    /// Save model checkpoint with versioning
    pub async fn save_checkpoint(
        &self,
        model: &dyn Model,
        checkpoint: ModelCheckpoint,
    ) -> Result<CheckpointId, StorageError> {
        let checkpoint_id = uuid::Uuid::new_v4().to_string();

        // Serialize model weights
        let temp_path = std::env::temp_dir().join(format!("{}.safetensors", checkpoint_id));
        model.save(&temp_path).await?;

        // Read weights data
        let weights_data = tokio::fs::read(&temp_path).await?;
        tokio::fs::remove_file(&temp_path).await?;

        // Store in AgentDB with vector embedding for similarity search
        let embedding = self.compute_embedding(&weights_data)?;

        let doc = Document::new()
            .set("id", &checkpoint_id)
            .set("model_id", model.metadata().id.as_str())
            .set("epoch", checkpoint.epoch as i64)
            .set("train_loss", checkpoint.train_loss)
            .set("val_loss", checkpoint.val_loss.unwrap_or(f64::NAN))
            .set("created_at", checkpoint.created_at.to_rfc3339())
            .set("weights_data", weights_data)
            .set_vector("embedding", embedding);

        self.checkpoints_collection.insert(doc).await?;
        Ok(CheckpointId(checkpoint_id))
    }

    /// Load model checkpoint
    pub async fn load_checkpoint(
        &self,
        checkpoint_id: &CheckpointId,
    ) -> Result<Box<dyn Model>, StorageError> {
        let doc = self.checkpoints_collection
            .get(&checkpoint_id.0)
            .await?
            .ok_or(StorageError::CheckpointNotFound)?;

        let model_id = doc.get_str("model_id")?;
        let model_metadata = self.load_model_metadata(model_id).await?;

        // Extract weights
        let weights_data = doc.get_bytes("weights_data")?;

        // Reconstruct model
        let temp_path = std::env::temp_dir().join(format!("{}.safetensors", checkpoint_id.0));
        tokio::fs::write(&temp_path, weights_data).await?;

        let model = self.reconstruct_model(&model_metadata, &temp_path).await?;
        tokio::fs::remove_file(&temp_path).await?;

        Ok(model)
    }

    /// Find similar models by architecture/performance
    pub async fn find_similar_models(
        &self,
        reference_model: &dyn Model,
        top_k: usize,
    ) -> Result<Vec<ModelId>, StorageError> {
        // Use AgentDB vector search to find similar model architectures
        let embedding = self.compute_model_embedding(reference_model)?;

        let similar = self.checkpoints_collection
            .search_vector(&embedding, top_k)
            .await?;

        Ok(similar.into_iter()
            .map(|doc| ModelId(doc.get_str("model_id").unwrap().to_string()))
            .collect())
    }

    /// List all models with filtering
    pub async fn list_models(
        &self,
        filter: ModelFilter,
    ) -> Result<Vec<ModelMetadata>, StorageError> {
        let query = filter.to_query();
        let results = self.models_collection.find(query).await?;

        results.into_iter()
            .map(|doc| self.doc_to_metadata(doc))
            .collect()
    }

    /// Version management
    pub async fn create_version(
        &self,
        model_id: &ModelId,
        version: &str,
        checkpoint_id: &CheckpointId,
    ) -> Result<(), StorageError> {
        let doc = Document::new()
            .set("model_id", model_id.0.as_str())
            .set("version", version)
            .set("checkpoint_id", checkpoint_id.0.as_str())
            .set("created_at", chrono::Utc::now().to_rfc3339());

        self.metadata_collection.insert(doc).await?;
        Ok(())
    }

    /// Compute embedding for checkpoint data
    fn compute_embedding(&self, weights_data: &[u8]) -> Result<Vec<f32>, StorageError> {
        // Use a hash-based or learned embedding for model weights
        // For simplicity, use a deterministic hash -> vector mapping
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        weights_data.hash(&mut hasher);
        let hash = hasher.finish();

        // Generate 128-dim embedding from hash
        let mut embedding = Vec::with_capacity(128);
        for i in 0..128 {
            let val = ((hash >> (i % 64)) & 0xFFFF) as f32 / 65535.0;
            embedding.push(val);
        }

        Ok(embedding)
    }

    /// Compute semantic embedding for model architecture
    fn compute_model_embedding(&self, model: &dyn Model) -> Result<Vec<f32>, StorageError> {
        let metadata = model.metadata();
        let input_spec = model.input_spec();
        let output_spec = model.output_spec();

        // Encode architecture features as embedding
        let mut features = vec![
            model.num_parameters() as f32 / 1e6,  // Normalize to millions
            model.memory_footprint() as f32 / 1e9, // Normalize to GB
            input_spec.shape.len() as f32,
            output_spec.shape.len() as f32,
        ];

        // Pad to 128 dimensions
        features.resize(128, 0.0);

        Ok(features)
    }

    fn reconstruct_model(
        &self,
        metadata: &ModelMetadata,
        weights_path: &Path,
    ) -> Result<Box<dyn Model>, StorageError> {
        // Build model from metadata and load weights
        todo!("Implement model reconstruction")
    }

    fn doc_to_metadata(&self, doc: Document) -> Result<ModelMetadata, StorageError> {
        todo!("Convert document to ModelMetadata")
    }

    async fn load_model_metadata(&self, model_id: &str) -> Result<ModelMetadata, StorageError> {
        todo!("Load metadata from DB")
    }
}

/// Model identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelId(pub String);

/// Checkpoint identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CheckpointId(pub String);

/// Model checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCheckpoint {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: Option<f64>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub metrics: HashMap<String, f64>,
}

/// Model filter for queries
#[derive(Debug, Default)]
pub struct ModelFilter {
    pub model_type: Option<ModelType>,
    pub framework: Option<Framework>,
    pub backend: Option<Backend>,
    pub tags: Vec<String>,
    pub min_params: Option<usize>,
    pub max_params: Option<usize>,
}

impl ModelFilter {
    fn to_query(&self) -> String {
        // Convert filter to AgentDB query syntax
        todo!("Implement query generation")
    }
}
```

### 4.2 Distributed Training Support

```rust
// nt-neural/src/distributed/coordinator.rs

use tokio::sync::{mpsc, RwLock};
use std::sync::Arc;

/// Distributed training coordinator
pub struct DistributedTrainer {
    registry: Arc<ModelRegistry>,
    workers: Vec<WorkerNode>,
    coordinator: TrainingCoordinator,
    config: DistributedConfig,
}

impl DistributedTrainer {
    pub async fn new(
        registry: Arc<ModelRegistry>,
        config: DistributedConfig,
    ) -> Result<Self, DistributedError> {
        let workers = Self::spawn_workers(&config).await?;
        let coordinator = TrainingCoordinator::new(workers.len());

        Ok(Self {
            registry,
            workers,
            coordinator,
            config,
        })
    }

    /// Train model across multiple nodes
    pub async fn train_distributed(
        &mut self,
        model: Box<dyn TrainableModel>,
        dataset: DistributedDataset,
        config: TrainingConfig,
    ) -> Result<Box<dyn Model>, DistributedError> {
        // Shard dataset across workers
        let shards = dataset.shard(self.workers.len())?;

        // Initialize model on each worker
        for (worker, shard) in self.workers.iter_mut().zip(shards) {
            worker.initialize_model(model.as_ref(), shard).await?;
        }

        // Training loop with synchronization
        for epoch in 0..config.num_epochs {
            // Forward/backward on each worker
            let mut futures = Vec::new();
            for worker in &mut self.workers {
                futures.push(worker.train_step(epoch));
            }

            let gradients = futures::future::try_join_all(futures).await?;

            // Aggregate gradients (AllReduce)
            let aggregated = self.coordinator.aggregate_gradients(gradients)?;

            // Broadcast updated parameters
            for worker in &mut self.workers {
                worker.update_parameters(&aggregated).await?;
            }

            // Checkpoint periodically
            if epoch % self.config.checkpoint_interval == 0 {
                self.save_checkpoint(epoch, &model).await?;
            }
        }

        // Collect final model from master worker
        let final_model = self.workers[0].collect_model().await?;
        Ok(final_model)
    }

    async fn spawn_workers(config: &DistributedConfig) -> Result<Vec<WorkerNode>, DistributedError> {
        let mut workers = Vec::new();

        for i in 0..config.num_workers {
            let worker = WorkerNode::spawn(i, &config.worker_config).await?;
            workers.push(worker);
        }

        Ok(workers)
    }

    async fn save_checkpoint(
        &self,
        epoch: usize,
        model: &Box<dyn TrainableModel>,
    ) -> Result<(), DistributedError> {
        let checkpoint = ModelCheckpoint {
            epoch,
            train_loss: 0.0, // TODO: Get from workers
            val_loss: None,
            created_at: chrono::Utc::now(),
            metrics: HashMap::new(),
        };

        self.registry.save_checkpoint(model.as_ref(), checkpoint).await?;
        Ok(())
    }
}

/// Worker node for distributed training
struct WorkerNode {
    id: usize,
    #[cfg(feature = "distributed")]
    process: tokio::process::Child,
    comm: mpsc::Sender<WorkerCommand>,
}

impl WorkerNode {
    async fn spawn(id: usize, config: &WorkerConfig) -> Result<Self, DistributedError> {
        #[cfg(feature = "distributed")]
        {
            // Spawn worker process (e.g., via MPI or custom protocol)
            todo!("Implement worker spawning")
        }

        #[cfg(not(feature = "distributed"))]
        {
            // In-process simulation for testing
            let (tx, _rx) = mpsc::channel(100);
            Ok(Self { id, comm: tx })
        }
    }

    async fn initialize_model(
        &mut self,
        model: &dyn TrainableModel,
        shard: DataShard,
    ) -> Result<(), DistributedError> {
        todo!("Send model + data to worker")
    }

    async fn train_step(&mut self, epoch: usize) -> Result<GradientUpdate, DistributedError> {
        todo!("Execute training step on worker")
    }

    async fn update_parameters(&mut self, gradients: &GradientUpdate) -> Result<(), DistributedError> {
        todo!("Apply gradient updates")
    }

    async fn collect_model(&mut self) -> Result<Box<dyn Model>, DistributedError> {
        todo!("Retrieve trained model")
    }
}

/// Training coordinator for gradient aggregation
struct TrainingCoordinator {
    num_workers: usize,
}

impl TrainingCoordinator {
    fn new(num_workers: usize) -> Self {
        Self { num_workers }
    }

    fn aggregate_gradients(&self, gradients: Vec<GradientUpdate>) -> Result<GradientUpdate, DistributedError> {
        // AllReduce: average gradients from all workers
        todo!("Implement gradient averaging")
    }
}

#[derive(Debug, Clone)]
pub struct DistributedConfig {
    pub num_workers: usize,
    pub checkpoint_interval: usize,
    pub worker_config: WorkerConfig,
    pub communication_backend: CommunicationBackend,
}

#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub gpu_id: Option<usize>,
    pub memory_limit_gb: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum CommunicationBackend {
    InProcess,  // For testing
    Tcp,        // TCP sockets
    Mpi,        // MPI (Message Passing Interface)
    Nccl,       // NVIDIA NCCL for multi-GPU
}

struct GradientUpdate {
    // Backend-specific gradient data
    data: Vec<u8>,
}

enum WorkerCommand {
    InitModel,
    TrainStep(usize),
    UpdateParams,
    Shutdown,
}
```

---

## 5. Multi-Backend Implementation Strategy

### 5.1 Backend Implementations

```rust
// nt-neural/src/backends/native.rs

/// Pure Rust CPU backend (no external dependencies)
pub struct NativeCpuBackend {
    // Use ndarray for matrix operations
}

impl BackendOps for NativeCpuBackend {
    fn create_tensor(&self, data: &[f64], shape: &[usize]) -> Result<TensorHandle, BackendError> {
        use ndarray::{Array, IxDyn};
        let array = Array::from_shape_vec(IxDyn(shape), data.to_vec())?;
        Ok(TensorHandle {
            inner: Box::new(array),
            shape: shape.to_vec(),
            dtype: DataType::F64,
            backend: Backend::CpuNative,
        })
    }

    fn matmul(&self, a: &TensorHandle, b: &TensorHandle) -> Result<TensorHandle, BackendError> {
        // Use ndarray's matrix multiplication
        todo!()
    }

    // ... implement all operations with ndarray/rayon
}

// nt-neural/src/backends/candle.rs

#[cfg(feature = "candle")]
pub struct CandleCpuBackend {
    device: candle_core::Device,
}

#[cfg(feature = "candle")]
impl BackendOps for CandleCpuBackend {
    fn create_tensor(&self, data: &[f64], shape: &[usize]) -> Result<TensorHandle, BackendError> {
        use candle_core::Tensor;
        let tensor = Tensor::from_vec(data.to_vec(), shape, &self.device)?;
        Ok(TensorHandle {
            inner: Box::new(tensor),
            shape: shape.to_vec(),
            dtype: DataType::F64,
            backend: Backend::Candle,
        })
    }

    // ... implement with candle operations
}

#[cfg(all(feature = "candle", feature = "cuda"))]
pub struct CandleCudaBackend {
    device: candle_core::Device,
}

#[cfg(all(feature = "candle", feature = "cuda"))]
impl BackendOps for CandleCudaBackend {
    // ... CUDA-accelerated implementations
}
```

### 5.2 Runtime Backend Selection

```rust
// nt-neural/src/core/runtime.rs

/// Runtime environment for neural operations
pub struct Runtime {
    backend: Arc<RwLock<Box<dyn BackendOps>>>,
    config: RuntimeConfig,
}

impl Runtime {
    /// Initialize runtime with optimal backend
    pub fn new(config: RuntimeConfig) -> Result<Self, RuntimeError> {
        let backend = match config.preferred_backend {
            Some(backend_type) => BackendFactory::create(backend_type)?,
            None => BackendFactory::create_auto()?,
        };

        Ok(Self {
            backend: Arc::new(RwLock::new(backend)),
            config,
        })
    }

    /// Get current backend
    pub async fn backend(&self) -> tokio::sync::RwLockReadGuard<'_, Box<dyn BackendOps>> {
        self.backend.read().await
    }

    /// Switch backend at runtime
    pub async fn switch_backend(&self, backend_type: Backend) -> Result<(), RuntimeError> {
        let new_backend = BackendFactory::create(backend_type)?;
        let mut backend_lock = self.backend.write().await;
        *backend_lock = new_backend;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub preferred_backend: Option<Backend>,
    pub enable_mixed_precision: bool,
    pub memory_limit_gb: Option<usize>,
}
```

---

## 6. Training Infrastructure Redesign

### 6.1 Unified Training Interface

```rust
// nt-neural/src/training/engine.rs

/// High-level training engine
pub struct TrainingEngine {
    runtime: Arc<Runtime>,
    registry: Arc<ModelRegistry>,
    callbacks: Vec<Box<dyn TrainingCallback>>,
}

impl TrainingEngine {
    pub fn new(runtime: Arc<Runtime>, registry: Arc<ModelRegistry>) -> Self {
        Self {
            runtime,
            registry,
            callbacks: Vec::new(),
        }
    }

    pub fn add_callback(&mut self, callback: Box<dyn TrainingCallback>) {
        self.callbacks.push(callback);
    }

    /// Train model with automatic checkpointing and monitoring
    pub async fn train(
        &mut self,
        mut model: Box<dyn TrainableModel>,
        dataset: TrainingDataset,
        config: TrainingConfig,
    ) -> Result<Box<dyn Model>, TrainingError> {
        // Register model in AgentDB
        let model_id = self.registry.register_model(model.as_ref(), vec![]).await?;

        // Create optimizer
        let mut optimizer = OptimizerFactory::create(&config.optimizer, &model)?;

        // Training loop
        for epoch in 0..config.num_epochs {
            // Callbacks: on_epoch_start
            for callback in &mut self.callbacks {
                callback.on_epoch_start(epoch, &model)?;
            }

            // Training step
            let train_metrics = self.train_epoch(&mut model, &dataset, &mut optimizer).await?;

            // Validation step
            let val_metrics = if let Some(val_dataset) = &dataset.validation {
                Some(self.validate_epoch(&model, val_dataset).await?)
            } else {
                None
            };

            // Callbacks: on_epoch_end
            for callback in &mut self.callbacks {
                callback.on_epoch_end(epoch, &model, &train_metrics, &val_metrics)?;
            }

            // Checkpoint
            if epoch % config.checkpoint_interval == 0 {
                let checkpoint = ModelCheckpoint {
                    epoch,
                    train_loss: train_metrics.loss,
                    val_loss: val_metrics.as_ref().map(|m| m.loss),
                    created_at: chrono::Utc::now(),
                    metrics: train_metrics.additional_metrics.clone(),
                };

                self.registry.save_checkpoint(model.as_ref(), checkpoint).await?;
            }
        }

        Ok(model)
    }

    async fn train_epoch(
        &self,
        model: &mut Box<dyn TrainableModel>,
        dataset: &TrainingDataset,
        optimizer: &mut Box<dyn Optimizer>,
    ) -> Result<EpochMetrics, TrainingError> {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        for batch in dataset.iter_batches() {
            // Forward pass
            let output = model.forward_train(&batch.input, &batch.target)?;

            // Backward pass
            model.backward(&output.loss, optimizer.as_mut())?;

            total_loss += output.loss.value();
            num_batches += 1;
        }

        Ok(EpochMetrics {
            loss: total_loss / num_batches as f64,
            additional_metrics: HashMap::new(),
        })
    }

    async fn validate_epoch(
        &self,
        model: &Box<dyn TrainableModel>,
        dataset: &ValidationDataset,
    ) -> Result<EpochMetrics, TrainingError> {
        // Similar to train_epoch but no backward pass
        todo!()
    }
}

/// Training callback trait
pub trait TrainingCallback: Send + Sync {
    fn on_epoch_start(&mut self, epoch: usize, model: &Box<dyn TrainableModel>) -> Result<(), TrainingError>;
    fn on_epoch_end(
        &mut self,
        epoch: usize,
        model: &Box<dyn TrainableModel>,
        train_metrics: &EpochMetrics,
        val_metrics: &Option<EpochMetrics>,
    ) -> Result<(), TrainingError>;
}

/// Early stopping callback
pub struct EarlyStoppingCallback {
    patience: usize,
    best_val_loss: Option<f64>,
    epochs_without_improvement: usize,
}

impl TrainingCallback for EarlyStoppingCallback {
    fn on_epoch_end(
        &mut self,
        epoch: usize,
        _model: &Box<dyn TrainableModel>,
        _train_metrics: &EpochMetrics,
        val_metrics: &Option<EpochMetrics>,
    ) -> Result<(), TrainingError> {
        if let Some(val_metrics) = val_metrics {
            if let Some(best) = self.best_val_loss {
                if val_metrics.loss < best {
                    self.best_val_loss = Some(val_metrics.loss);
                    self.epochs_without_improvement = 0;
                } else {
                    self.epochs_without_improvement += 1;
                    if self.epochs_without_improvement >= self.patience {
                        return Err(TrainingError::EarlyStopping(epoch));
                    }
                }
            } else {
                self.best_val_loss = Some(val_metrics.loss);
            }
        }
        Ok(())
    }

    fn on_epoch_start(&mut self, _epoch: usize, _model: &Box<dyn TrainableModel>) -> Result<(), TrainingError> {
        Ok(())
    }
}

struct EpochMetrics {
    loss: f64,
    additional_metrics: HashMap<String, f64>,
}
```

---

## 7. Inference Optimization

### 7.1 Fast Inference Pipeline

```rust
// nt-neural/src/inference/fast.rs

/// Ultra-fast inference engine with <10ms latency
pub struct FastInferenceEngine {
    runtime: Arc<Runtime>,
    model_cache: Arc<RwLock<ModelCache>>,
    config: InferenceConfig,
}

impl FastInferenceEngine {
    pub fn new(runtime: Arc<Runtime>, config: InferenceConfig) -> Self {
        Self {
            runtime,
            model_cache: Arc::new(RwLock::new(ModelCache::new(config.cache_size))),
            config,
        }
    }

    /// Single prediction with caching
    pub async fn predict(
        &self,
        model_id: &ModelId,
        input: &InputData,
    ) -> Result<PredictionOutput, InferenceError> {
        let start = Instant::now();

        // Get model from cache
        let model = self.get_or_load_model(model_id).await?;

        // Predict
        let output = model.forward(input)?;

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        tracing::debug!("Inference completed in {:.2}ms", latency_ms);

        Ok(output)
    }

    /// Batch prediction with optimal batching
    pub async fn predict_batch(
        &self,
        model_id: &ModelId,
        inputs: Vec<InputData>,
    ) -> Result<Vec<PredictionOutput>, InferenceError> {
        let model = self.get_or_load_model(model_id).await?;

        // Dynamic batching based on GPU memory
        let optimal_batch_size = self.compute_optimal_batch_size(&model, &inputs)?;

        let mut outputs = Vec::with_capacity(inputs.len());
        for chunk in inputs.chunks(optimal_batch_size) {
            let batch_outputs = model.forward_batch(chunk)?;
            outputs.extend(batch_outputs);
        }

        Ok(outputs)
    }

    async fn get_or_load_model(&self, model_id: &ModelId) -> Result<Arc<Box<dyn InferenceModel>>, InferenceError> {
        // Check cache
        {
            let cache = self.model_cache.read().await;
            if let Some(model) = cache.get(model_id) {
                return Ok(Arc::clone(model));
            }
        }

        // Load from AgentDB
        // TODO: inject registry
        // let model = self.registry.load_model(model_id).await?;

        // Cache model
        // let mut cache = self.model_cache.write().await;
        // cache.insert(model_id.clone(), Arc::new(model));

        todo!("Implement model loading")
    }

    fn compute_optimal_batch_size(
        &self,
        model: &Arc<Box<dyn InferenceModel>>,
        inputs: &[InputData],
    ) -> Result<usize, InferenceError> {
        // Estimate based on model memory footprint and available GPU memory
        let model_memory_gb = model.memory_footprint() as f64 / 1e9;
        let available_memory_gb = self.get_available_memory()?;

        let estimated_batch_size = ((available_memory_gb / model_memory_gb) * 0.8) as usize;
        Ok(estimated_batch_size.max(1).min(inputs.len()))
    }

    fn get_available_memory(&self) -> Result<f64, InferenceError> {
        // Query GPU memory if available
        #[cfg(feature = "cuda")]
        {
            // Use cudarc to query memory
            todo!()
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Fallback to system memory
            Ok(16.0) // Assume 16GB
        }
    }
}

/// Model cache with LRU eviction
struct ModelCache {
    cache: lru::LruCache<ModelId, Arc<Box<dyn InferenceModel>>>,
}

impl ModelCache {
    fn new(capacity: usize) -> Self {
        Self {
            cache: lru::LruCache::new(capacity.try_into().unwrap()),
        }
    }

    fn get(&mut self, key: &ModelId) -> Option<&Arc<Box<dyn InferenceModel>>> {
        self.cache.get(key)
    }

    fn insert(&mut self, key: ModelId, value: Arc<Box<dyn InferenceModel>>) {
        self.cache.put(key, value);
    }
}

#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub cache_size: usize,
    pub enable_batching: bool,
    pub max_batch_size: usize,
    pub timeout_ms: u64,
}
```

---

## 8. Migration Path from Current Implementation

### 8.1 Phase 1: Core Abstractions (Week 1-2)

**Tasks:**
1. Implement `Model`, `TrainableModel`, `InferenceModel` traits
2. Create `BackendOps` trait and `BackendFactory`
3. Implement `NativeCpuBackend` (ndarray-based)
4. Add `CandleCpuBackend` wrapper around existing code
5. Create `ModelBuilder` pattern
6. Update error types with new variants

**Deliverables:**
- `/workspaces/neural-trader/neural-trader-rust/crates/neural/src/core/`
  - `model.rs` - Model traits
  - `backend.rs` - Backend abstraction
  - `builder.rs` - Builder pattern
  - `runtime.rs` - Runtime environment

### 8.2 Phase 2: AgentDB Integration (Week 3-4)

**Tasks:**
1. Design AgentDB schema for models/checkpoints
2. Implement `ModelRegistry` with AgentDB storage
3. Add vector embeddings for model similarity search
4. Create version management system
5. Implement checkpoint save/load with AgentDB

**Deliverables:**
- `/workspaces/neural-trader/neural-trader-rust/crates/neural/src/storage/`
  - `agentdb.rs` - AgentDB integration
  - `registry.rs` - Model registry
  - `versioning.rs` - Version management

### 8.3 Phase 3: Distributed Training (Week 5-6)

**Tasks:**
1. Implement `DistributedTrainer`
2. Create worker node protocol
3. Add gradient aggregation (AllReduce)
4. Implement data sharding
5. Add distributed checkpointing

**Deliverables:**
- `/workspaces/neural-trader/neural-trader-rust/crates/neural/src/distributed/`
  - `coordinator.rs` - Training coordinator
  - `worker.rs` - Worker node
  - `communication.rs` - Inter-process communication

### 8.4 Phase 4: Model Migration (Week 7-8)

**Tasks:**
1. Refactor `NHITSModel` to use new traits
2. Refactor `LSTMAttentionModel`
3. Refactor `TransformerModel`
4. Update training code to use `TrainingEngine`
5. Update inference code to use `FastInferenceEngine`
6. Add integration tests

**Deliverables:**
- Refactored models in `/workspaces/neural-trader/neural-trader-rust/crates/neural/src/models/`
- Updated training in `/workspaces/neural-trader/neural-trader-rust/crates/neural/src/training/`
- Updated inference in `/workspaces/neural-trader/neural-trader-rust/crates/neural/src/inference/`

### 8.5 Phase 5: Documentation & Testing (Week 9-10)

**Tasks:**
1. Write comprehensive API documentation
2. Create usage examples
3. Add benchmarks for all backends
4. Performance testing (inference <10ms)
5. Integration testing with AgentDB
6. Distributed training tests

**Deliverables:**
- `/workspaces/neural-trader/docs/neural/`
  - `USER_GUIDE.md`
  - `API_REFERENCE.md`
  - `EXAMPLES.md`
  - `BENCHMARKS.md`

---

## 9. Feature Gate Redesign

### 9.1 Updated Feature Flags

```toml
# Cargo.toml

[features]
default = ["native-cpu"]

# Backend selection (mutually exclusive at runtime, not compile time)
native-cpu = []                              # Pure Rust CPU backend
candle = ["dep:candle-core", "dep:candle-nn"] # Candle framework
burn = ["dep:burn"]                          # Burn framework (future)
smartcore = ["dep:smartcore"]                # SmartCore (future)

# GPU acceleration (requires candle)
cuda = ["candle", "dep:cudarc", "candle-core/cuda"]
metal = ["candle", "candle-core/metal"]
accelerate = ["candle", "candle-core/accelerate"]

# Storage & distributed
agentdb = ["dep:agentdb"]
distributed = ["dep:mpi", "dep:nccl"]        # Distributed training

# Optimizations
quantization = []                             # INT8/FP16 quantization
simd = []                                     # SIMD optimizations

# Development
benchmarks = []
testing = []
```

### 9.2 Conditional Compilation Strategy

```rust
// Compile-time backend selection
#[cfg(feature = "candle")]
type DefaultBackend = CandleCpuBackend;

#[cfg(all(not(feature = "candle"), feature = "native-cpu"))]
type DefaultBackend = NativeCpuBackend;

#[cfg(all(not(feature = "candle"), not(feature = "native-cpu")))]
compile_error!("At least one backend must be enabled");

// Runtime backend selection (when multiple backends compiled)
pub fn create_optimal_backend() -> Box<dyn BackendOps> {
    #[cfg(feature = "cuda")]
    if let Ok(backend) = CandleCudaBackend::new(0) {
        return Box::new(backend);
    }

    #[cfg(feature = "metal")]
    if let Ok(backend) = CandleMetalBackend::new(0) {
        return Box::new(backend);
    }

    #[cfg(feature = "candle")]
    return Box::new(CandleCpuBackend::new());

    #[cfg(feature = "native-cpu")]
    return Box::new(NativeCpuBackend::new());
}
```

---

## 10. Performance Targets

### 10.1 Inference Latency

| Backend | Target Latency | Max Throughput |
|---------|---------------|----------------|
| Native CPU | <50ms | 20 req/s |
| Candle CPU | <20ms | 50 req/s |
| Candle CUDA | <5ms | 200 req/s |
| Candle Metal | <10ms | 100 req/s |

### 10.2 Training Performance

| Configuration | Target Speed | Memory Usage |
|--------------|--------------|--------------|
| Single CPU | 1x baseline | <4GB |
| Single GPU (CUDA) | 10x baseline | <8GB |
| 4-GPU Distributed | 35x baseline | <32GB |
| 8-GPU Distributed | 60x baseline | <64GB |

### 10.3 Storage Efficiency

- **Model Compression**: 10x via quantization (FP32 → INT8)
- **AgentDB Overhead**: <5% over raw safetensors
- **Checkpoint Size**: <500MB per checkpoint (NHITS model)

---

## 11. Testing Strategy

### 11.1 Unit Tests

```rust
// nt-neural/src/core/model.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_model_builder() {
        let config = NHITSConfig::default();
        let model = ModelBuilder::new(ModelType::NHITS)
            .with_config(config)
            .build()
            .unwrap();

        assert_eq!(model.model_type(), ModelType::NHITS);
    }

    #[tokio::test]
    async fn test_backend_switching() {
        let runtime = Runtime::new(RuntimeConfig::default()).unwrap();

        // Start with CPU
        assert_eq!(runtime.backend().await.device(), BackendDevice::Cpu);

        // Switch to GPU (if available)
        #[cfg(feature = "cuda")]
        {
            runtime.switch_backend(Backend::CandleCuda).await.unwrap();
            assert_eq!(runtime.backend().await.device(), BackendDevice::Cuda(0));
        }
    }
}
```

### 11.2 Integration Tests

```rust
// nt-neural/tests/integration_test.rs

#[tokio::test]
async fn test_full_training_pipeline() {
    // Setup
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let registry = ModelRegistry::new("test_db.agentdb").await.unwrap();
    let mut engine = TrainingEngine::new(Arc::new(runtime), Arc::new(registry));

    // Create model
    let model = ModelBuilder::new(ModelType::NHITS)
        .with_config(NHITSConfig::default())
        .build()
        .unwrap();

    // Create dataset
    let dataset = TrainingDataset::from_csv("test_data.csv").await.unwrap();

    // Train
    let config = TrainingConfig {
        num_epochs: 10,
        batch_size: 32,
        ..Default::default()
    };

    let trained_model = engine.train(model, dataset, config).await.unwrap();

    // Verify checkpoint saved
    let checkpoints = registry.list_checkpoints(&trained_model.metadata().id).await.unwrap();
    assert!(!checkpoints.is_empty());
}
```

### 11.3 Benchmark Tests

```rust
// nt-neural/benches/inference_bench.rs

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_inference_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference_latency");

    for backend in &[Backend::CpuNative, Backend::Candle] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", backend)),
            backend,
            |b, backend| {
                let model = create_test_model(*backend);
                let input = create_test_input();

                b.iter(|| {
                    model.forward(&input).unwrap()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_inference_latency);
criterion_main!(benches);
```

---

## 12. Module Organization Chart

```
nt-neural/
├── src/
│   ├── lib.rs                      # Public API
│   ├── error.rs                    # Error types
│   │
│   ├── core/                       # Core abstractions
│   │   ├── mod.rs
│   │   ├── model.rs                # Model traits
│   │   ├── backend.rs              # Backend abstraction
│   │   ├── builder.rs              # Builder pattern
│   │   ├── runtime.rs              # Runtime environment
│   │   └── types.rs                # Common types
│   │
│   ├── backends/                   # Backend implementations
│   │   ├── mod.rs
│   │   ├── native.rs               # ndarray-based CPU
│   │   ├── candle.rs               # Candle CPU/GPU
│   │   ├── burn.rs                 # Burn (future)
│   │   └── smartcore.rs            # SmartCore (future)
│   │
│   ├── models/                     # Model architectures
│   │   ├── mod.rs
│   │   ├── nhits.rs
│   │   ├── lstm_attention.rs
│   │   ├── transformer.rs
│   │   ├── layers.rs               # Reusable layers
│   │   └── registry.rs             # Model registry
│   │
│   ├── training/                   # Training infrastructure
│   │   ├── mod.rs
│   │   ├── engine.rs               # High-level training
│   │   ├── trainer.rs              # Training loop
│   │   ├── optimizer.rs            # Optimizers
│   │   ├── scheduler.rs            # LR schedulers
│   │   ├── data_loader.rs          # Data loading
│   │   ├── callbacks.rs            # Training callbacks
│   │   └── metrics.rs              # Training metrics
│   │
│   ├── inference/                  # Inference engines
│   │   ├── mod.rs
│   │   ├── fast.rs                 # <10ms inference
│   │   ├── batch.rs                # Batch inference
│   │   ├── streaming.rs            # Streaming inference
│   │   └── cache.rs                # Model caching
│   │
│   ├── storage/                    # Model persistence
│   │   ├── mod.rs
│   │   ├── agentdb.rs              # AgentDB integration
│   │   ├── registry.rs             # Model registry
│   │   ├── versioning.rs           # Version management
│   │   └── checkpoint.rs           # Checkpointing
│   │
│   ├── distributed/                # Distributed training
│   │   ├── mod.rs
│   │   ├── coordinator.rs          # Training coordinator
│   │   ├── worker.rs               # Worker nodes
│   │   ├── communication.rs        # Inter-process comm
│   │   └── sharding.rs             # Data sharding
│   │
│   └── utils/                      # Utilities
│       ├── mod.rs
│       ├── preprocessing.rs
│       ├── metrics.rs
│       ├── validation.rs
│       └── features.rs
│
├── tests/                          # Integration tests
│   ├── integration_test.rs
│   ├── backend_test.rs
│   └── distributed_test.rs
│
├── benches/                        # Benchmarks
│   ├── inference_bench.rs
│   └── training_bench.rs
│
└── examples/                       # Usage examples
    ├── simple_training.rs
    ├── distributed_training.rs
    └── inference_server.rs
```

---

## 13. API Examples

### 13.1 Basic Training

```rust
use nt_neural::{ModelBuilder, ModelType, NHITSConfig, TrainingEngine, TrainingConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize runtime
    let runtime = Runtime::new(RuntimeConfig::default())?;
    let registry = ModelRegistry::new("models.agentdb").await?;

    // Create model
    let config = NHITSConfig::default();
    let model = ModelBuilder::new(ModelType::NHITS)
        .with_config(config)
        .build()?;

    // Load dataset
    let dataset = TrainingDataset::from_csv("data.csv").await?;

    // Train
    let mut engine = TrainingEngine::new(Arc::new(runtime), Arc::new(registry));
    engine.add_callback(Box::new(EarlyStoppingCallback::new(10)));

    let trained_model = engine.train(
        model,
        dataset,
        TrainingConfig {
            num_epochs: 100,
            batch_size: 32,
            learning_rate: 1e-3,
            ..Default::default()
        },
    ).await?;

    println!("Training complete!");
    Ok(())
}
```

### 13.2 Fast Inference

```rust
use nt_neural::{FastInferenceEngine, InferenceConfig, ModelId};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let runtime = Runtime::new(RuntimeConfig::default())?;
    let engine = FastInferenceEngine::new(
        Arc::new(runtime),
        InferenceConfig {
            cache_size: 10,
            enable_batching: true,
            max_batch_size: 128,
            timeout_ms: 100,
        },
    );

    let model_id = ModelId("nhits-v1".to_string());
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let prediction = engine.predict(&model_id, &input).await?;
    println!("Prediction: {:?}", prediction);

    Ok(())
}
```

### 13.3 Distributed Training

```rust
use nt_neural::{DistributedTrainer, DistributedConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let registry = Arc::new(ModelRegistry::new("models.agentdb").await?);

    let config = DistributedConfig {
        num_workers: 4,
        checkpoint_interval: 10,
        worker_config: WorkerConfig {
            gpu_id: Some(0),
            memory_limit_gb: 16,
        },
        communication_backend: CommunicationBackend::Nccl,
    };

    let mut trainer = DistributedTrainer::new(registry, config).await?;

    let model = ModelBuilder::new(ModelType::NHITS)
        .with_config(NHITSConfig::default())
        .build()?;

    let dataset = DistributedDataset::from_csv("large_data.csv").await?;

    let trained_model = trainer.train_distributed(
        model,
        dataset,
        TrainingConfig::default(),
    ).await?;

    println!("Distributed training complete!");
    Ok(())
}
```

---

## 14. Appendix

### 14.1 Glossary

- **Backend**: The underlying computational framework (Candle, native CPU, etc.)
- **AgentDB**: Vector database for model storage and similarity search
- **Checkpoint**: Saved model state during training
- **Distributed Training**: Training across multiple machines/GPUs
- **Model Registry**: Centralized repository for model discovery
- **Quantization**: Reducing model precision (FP32 → INT8) for efficiency

### 14.2 References

- [Candle Documentation](https://github.com/huggingface/candle)
- [AgentDB Documentation](https://github.com/agentdb/agentdb)
- [NHITS Paper](https://arxiv.org/abs/2201.12886)
- [Burn ML Framework](https://github.com/burn-rs/burn)

### 14.3 Decision Records

#### ADR-001: Backend Abstraction Layer

**Context**: Need to support multiple ML frameworks (Candle, Burn, native) without code duplication.

**Decision**: Implement a trait-based backend abstraction (`BackendOps`) that allows runtime backend selection.

**Consequences**:
- ✅ Framework flexibility
- ✅ Testing with mock backends
- ⚠️ Minor runtime overhead from trait dispatch
- ⚠️ Increased complexity

#### ADR-002: AgentDB for Model Storage

**Context**: Need versioning, similarity search, and distributed access to models.

**Decision**: Use AgentDB as the model registry and checkpoint storage.

**Consequences**:
- ✅ Vector search for model similarity
- ✅ Built-in versioning
- ✅ Distributed access
- ⚠️ Additional dependency
- ⚠️ Storage overhead (~5%)

#### ADR-003: Async Training API

**Context**: Training involves I/O (checkpointing, data loading).

**Decision**: Make training APIs async using Tokio.

**Consequences**:
- ✅ Non-blocking I/O
- ✅ Better resource utilization
- ⚠️ Increased API complexity
- ⚠️ Async runtime required

---

## 15. Next Steps

### Immediate Actions

1. **Review & Approval**: Team review of this architecture document
2. **Proof of Concept**: Implement Phase 1 (Core Abstractions) for validation
3. **Benchmark Baseline**: Establish performance baselines for comparison
4. **AgentDB Integration Test**: Verify AgentDB can handle model storage requirements

### Future Considerations

1. **Model Quantization**: INT8/FP16 for 10x model compression
2. **ONNX Export**: Enable deployment to non-Rust environments
3. **Model Serving**: REST API server for production inference
4. **AutoML**: Hyperparameter tuning with Optuna/Hyperopt
5. **Explainability**: SHAP/LIME for model interpretation

---

**Document Status**: ✅ Complete
**Author**: System Architect
**Last Updated**: 2025-11-13
**Version**: 1.0.0
