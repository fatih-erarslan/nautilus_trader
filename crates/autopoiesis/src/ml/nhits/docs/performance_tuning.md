# Performance Tuning Guide

Comprehensive guide for optimizing NHITS performance across different dimensions: computational efficiency, memory usage, throughput, and latency.

## Table of Contents

- [Performance Overview](#performance-overview)
- [Model Architecture Optimization](#model-architecture-optimization)
- [Memory Optimization](#memory-optimization)
- [CPU Optimization](#cpu-optimization)
- [I/O Optimization](#io-optimization)
- [Distributed Computing](#distributed-computing)
- [GPU Acceleration](#gpu-acceleration)
- [Caching Strategies](#caching-strategies)
- [Monitoring & Profiling](#monitoring--profiling)
- [Benchmarking](#benchmarking)

## Performance Overview

### Key Performance Metrics

| Metric | Target | Good | Acceptable |
|--------|--------|------|------------|
| Training Speed | 10K+ samples/sec | 5K-10K | 1K-5K |
| Inference Speed | 100K+ pred/sec | 50K-100K | 10K-50K |
| Memory Usage | < 1GB/1M steps | 1-2GB | 2-4GB |
| Forecast Latency | < 10ms | 10-50ms | 50-200ms |
| Consciousness Coherence | > 0.95 | 0.90-0.95 | 0.80-0.90 |

### Performance Characteristics by Configuration

```rust
// High-performance configuration
let high_perf_config = NHITSConfigBuilder::new()
    .with_lookback(50)          // Shorter sequences
    .with_horizon(10)           // Shorter horizons
    .with_blocks(vec![
        BlockConfig {
            hidden_size: 256,   // Moderate size
            num_basis: 5,       // Fewer basis functions
            pooling_factor: 4,  // Aggressive pooling
            activation: ActivationType::ReLU, // Fast activation
            dropout_rate: 0.0,  // No dropout for inference
            ..Default::default()
        }
    ])
    .with_consciousness(false, 0.0) // Disable for max speed
    .build()?;

// Balanced configuration
let balanced_config = NHITSConfigBuilder::new()
    .with_lookback(168)
    .with_horizon(24)
    .with_blocks(vec![
        BlockConfig {
            hidden_size: 512,
            num_basis: 8,
            pooling_factor: 2,
            activation: ActivationType::GELU,
            dropout_rate: 0.1,
            ..Default::default()
        }
    ])
    .with_consciousness(true, 0.1)
    .build()?;

// High-accuracy configuration
let accuracy_config = NHITSConfigBuilder::new()
    .with_lookbook(720)         // Long sequences
    .with_horizon(168)          // Long horizons
    .with_blocks(vec![
        BlockConfig {
            hidden_size: 1024,  // Large hidden size
            num_basis: 15,      // Many basis functions
            pooling_factor: 2,  // Conservative pooling
            activation: ActivationType::GELU,
            dropout_rate: 0.2,
            ..Default::default()
        },
        // Multiple blocks for complexity
        BlockConfig { hidden_size: 512, ..Default::default() },
        BlockConfig { hidden_size: 256, ..Default::default() },
    ])
    .with_consciousness(true, 0.3) // High consciousness weight
    .build()?;
```

## Model Architecture Optimization

### Block Configuration Tuning

```rust
// Optimize number of blocks vs. performance
fn optimize_block_count(data: &Array3<f64>) -> Vec<BlockConfig> {
    let data_complexity = estimate_complexity(data);
    
    match data_complexity {
        Complexity::Low => vec![
            BlockConfig {
                hidden_size: 128,
                num_basis: 4,
                pooling_factor: 4,
                ..Default::default()
            }
        ],
        Complexity::Medium => vec![
            BlockConfig { hidden_size: 256, num_basis: 6, ..Default::default() },
            BlockConfig { hidden_size: 128, num_basis: 4, ..Default::default() },
        ],
        Complexity::High => vec![
            BlockConfig { hidden_size: 512, num_basis: 10, ..Default::default() },
            BlockConfig { hidden_size: 256, num_basis: 8, ..Default::default() },
            BlockConfig { hidden_size: 128, num_basis: 6, ..Default::default() },
        ],
    }
}

fn estimate_complexity(data: &Array3<f64>) -> Complexity {
    let variance = data.var(0.0);
    let entropy = calculate_entropy(data);
    let trend_strength = calculate_trend_strength(data);
    
    let complexity_score = variance * 0.3 + entropy * 0.4 + trend_strength * 0.3;
    
    if complexity_score < 0.3 { Complexity::Low }
    else if complexity_score < 0.7 { Complexity::Medium }
    else { Complexity::High }
}
```

### Attention Optimization

```rust
// Optimize attention configuration for different scenarios
fn optimize_attention_config(use_case: UseCase, seq_len: usize) -> AttentionConfig {
    match use_case {
        UseCase::HighFrequencyTrading => {
            AttentionConfig {
                num_heads: 4,               // Fewer heads for speed
                head_dim: 32,               // Smaller dimensions
                attention_type: AttentionType::LocalWindow { window_size: 10 },
                dropout_rate: 0.0,          // No dropout for inference
                consciousness_integration: false, // Disable for max speed
                ..Default::default()
            }
        },
        UseCase::LongTermForecasting => {
            AttentionConfig {
                num_heads: 8,
                head_dim: 64,
                attention_type: AttentionType::Sparse { sparsity_factor: 0.9 },
                dropout_rate: 0.1,
                consciousness_integration: true,
                ..Default::default()
            }
        },
        _ => AttentionConfig::default(),
    }
}
```

### Adaptive Structure Tuning

```rust
// Configure adaptation for optimal performance
fn optimize_adaptation_config(performance_requirements: &PerformanceRequirements) -> AdaptationConfig {
    AdaptationConfig {
        adaptation_rate: if performance_requirements.low_latency { 0.001 } else { 0.01 },
        performance_window: if performance_requirements.stability { 50 } else { 20 },
        change_threshold: if performance_requirements.stability { 0.001 } else { 0.01 },
        adaptation_strategy: if performance_requirements.accuracy_first {
            AdaptationStrategy::ConsciousnessGuided
        } else {
            AdaptationStrategy::Conservative
        },
        exploration_rate: if performance_requirements.low_latency { 0.01 } else { 0.1 },
        ..Default::default()
    }
}
```

## Memory Optimization

### Memory-Efficient Data Structures

```rust
use ndarray::{Array3, ArrayView3};
use std::sync::Arc;

// Use memory-mapped arrays for large datasets
struct MemoryEfficientNHITS {
    model: NHITS,
    data_cache: Arc<MemoryMappedArray>,
    gradient_cache: Option<Array3<f32>>, // Use f32 instead of f64 when possible
}

impl MemoryEfficientNHITS {
    // Streaming forward pass to reduce memory footprint
    pub fn forward_streaming(
        &mut self,
        input: ArrayView3<f64>,
        chunk_size: usize,
    ) -> Result<Array3<f64>, NHITSError> {
        let total_samples = input.shape()[0];
        let mut results = Vec::new();
        
        for chunk_start in (0..total_samples).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(total_samples);
            let chunk = input.slice(s![chunk_start..chunk_end, .., ..]);
            
            // Process chunk
            let chunk_result = self.model.forward(&chunk.to_owned(), 
                self.model.config.lookback_window, 
                self.model.config.forecast_horizon)?;
            
            results.push(chunk_result);
            
            // Clear intermediate computations to free memory
            self.clear_cache();
        }
        
        // Concatenate results
        let combined = concatenate_arrays(&results)?;
        Ok(combined)
    }
    
    fn clear_cache(&mut self) {
        self.gradient_cache = None;
        // Force garbage collection of intermediate tensors
        std::mem::drop(std::mem::take(&mut self.gradient_cache));
    }
}
```

### Memory Pool Management

```rust
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;

// Memory pool for tensor reuse
pub struct TensorPool {
    f64_pool: Arc<Mutex<VecDeque<Array3<f64>>>>,
    f32_pool: Arc<Mutex<VecDeque<Array3<f32>>>>,
}

impl TensorPool {
    pub fn new(initial_capacity: usize) -> Self {
        let mut f64_pool = VecDeque::with_capacity(initial_capacity);
        let mut f32_pool = VecDeque::with_capacity(initial_capacity);
        
        // Pre-allocate common tensor sizes
        for _ in 0..initial_capacity {
            f64_pool.push_back(Array3::zeros((32, 168, 1)));
            f32_pool.push_back(Array3::zeros((32, 168, 1)));
        }
        
        Self {
            f64_pool: Arc::new(Mutex::new(f64_pool)),
            f32_pool: Arc::new(Mutex::new(f32_pool)),
        }
    }
    
    pub fn get_f64_tensor(&self, shape: (usize, usize, usize)) -> Array3<f64> {
        let mut pool = self.f64_pool.lock().unwrap();
        
        if let Some(mut tensor) = pool.pop_front() {
            if tensor.shape() == &[shape.0, shape.1, shape.2] {
                tensor.fill(0.0);
                return tensor;
            }
        }
        
        Array3::zeros(shape)
    }
    
    pub fn return_f64_tensor(&self, tensor: Array3<f64>) {
        let mut pool = self.f64_pool.lock().unwrap();
        if pool.len() < 100 { // Max pool size
            pool.push_back(tensor);
        }
    }
}
```

### Memory Usage Monitoring

```rust
use sysinfo::{System, SystemExt};

pub struct MemoryMonitor {
    system: System,
    peak_usage: usize,
    allocation_tracker: HashMap<String, usize>,
}

impl MemoryMonitor {
    pub fn new() -> Self {
        Self {
            system: System::new_all(),
            peak_usage: 0,
            allocation_tracker: HashMap::new(),
        }
    }
    
    pub fn track_allocation(&mut self, name: &str, size: usize) {
        self.allocation_tracker.insert(name.to_string(), size);
        self.update_peak_usage();
    }
    
    pub fn get_memory_stats(&mut self) -> MemoryStats {
        self.system.refresh_memory();
        
        let current_usage = self.system.used_memory() as usize;
        let available = self.system.available_memory() as usize;
        
        MemoryStats {
            current_usage,
            peak_usage: self.peak_usage,
            available,
            allocations: self.allocation_tracker.clone(),
        }
    }
    
    fn update_peak_usage(&mut self) {
        self.system.refresh_memory();
        let current = self.system.used_memory() as usize;
        if current > self.peak_usage {
            self.peak_usage = current;
        }
    }
}
```

## CPU Optimization

### SIMD Vectorization

```rust
use std::arch::x86_64::*;

// SIMD-optimized matrix operations
pub struct SIMDOptimizedOps;

impl SIMDOptimizedOps {
    #[target_feature(enable = "avx2")]
    pub unsafe fn vectorized_multiply(a: &[f64], b: &[f64], result: &mut [f64]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());
        
        let len = a.len();
        let simd_len = len - (len % 4);
        
        // Process 4 elements at a time
        for i in (0..simd_len).step_by(4) {
            let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
            let b_vec = _mm256_loadu_pd(b.as_ptr().add(i));
            let result_vec = _mm256_mul_pd(a_vec, b_vec);
            _mm256_storeu_pd(result.as_mut_ptr().add(i), result_vec);
        }
        
        // Handle remaining elements
        for i in simd_len..len {
            result[i] = a[i] * b[i];
        }
    }
    
    #[target_feature(enable = "avx2")]
    pub unsafe fn vectorized_relu(input: &[f64], output: &mut [f64]) {
        let zero = _mm256_setzero_pd();
        let len = input.len();
        let simd_len = len - (len % 4);
        
        for i in (0..simd_len).step_by(4) {
            let x = _mm256_loadu_pd(input.as_ptr().add(i));
            let result = _mm256_max_pd(x, zero);
            _mm256_storeu_pd(output.as_mut_ptr().add(i), result);
        }
        
        for i in simd_len..len {
            output[i] = input[i].max(0.0);
        }
    }
}
```

### Parallel Processing

```rust
use rayon::prelude::*;
use std::sync::Arc;

// Parallel ensemble inference
pub struct ParallelEnsemble {
    models: Vec<NHITS>,
    thread_pool: rayon::ThreadPool,
}

impl ParallelEnsemble {
    pub fn new(models: Vec<NHITS>, num_threads: usize) -> Self {
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("Failed to create thread pool");
        
        Self { models, thread_pool }
    }
    
    pub fn parallel_forecast(
        &self,
        input: &Array3<f64>,
        lookback: usize,
        horizon: usize,
    ) -> Result<Vec<Array3<f64>>, NHITSError> {
        let input = Arc::new(input.clone());
        
        let results: Result<Vec<_>, _> = self.thread_pool.install(|| {
            self.models
                .par_iter()
                .map(|model| {
                    let input_clone = input.clone();
                    let mut model_clone = model.clone();
                    model_clone.forward(&input_clone, lookback, horizon)
                })
                .collect()
        });
        
        results
    }
    
    pub fn aggregate_predictions(
        &self,
        predictions: &[Array3<f64>],
        weights: Option<&[f64]>,
    ) -> Array3<f64> {
        let shape = predictions[0].shape();
        let mut result = Array3::zeros(shape);
        
        if let Some(w) = weights {
            predictions
                .par_iter()
                .zip(w.par_iter())
                .map(|(pred, &weight)| pred * weight)
                .reduce(|| Array3::zeros(shape), |acc, pred| acc + pred)
        } else {
            let weight = 1.0 / predictions.len() as f64;
            predictions
                .par_iter()
                .map(|pred| pred * weight)
                .reduce(|| Array3::zeros(shape), |acc, pred| acc + pred)
        }
    }
}
```

### CPU Affinity and NUMA Optimization

```rust
use std::thread;
use libc::{cpu_set_t, sched_setaffinity, CPU_SET, CPU_ZERO};

pub struct CPUOptimizer;

impl CPUOptimizer {
    pub fn pin_to_core(core_id: usize) -> Result<(), std::io::Error> {
        unsafe {
            let mut cpuset: cpu_set_t = std::mem::zeroed();
            CPU_ZERO(&mut cpuset);
            CPU_SET(core_id, &mut cpuset);
            
            let result = sched_setaffinity(0, std::mem::size_of::<cpu_set_t>(), &cpuset);
            if result != 0 {
                return Err(std::io::Error::last_os_error());
            }
        }
        Ok(())
    }
    
    pub fn optimize_for_numa() -> Result<(), Box<dyn std::error::Error>> {
        // Get NUMA topology
        let numa_nodes = Self::get_numa_nodes()?;
        let current_node = Self::get_current_numa_node()?;
        
        // Allocate memory on local NUMA node
        Self::set_memory_policy(current_node)?;
        
        // Pin threads to local cores
        let local_cores = Self::get_cores_for_numa_node(current_node)?;
        Self::distribute_threads_on_cores(&local_cores)?;
        
        Ok(())
    }
    
    fn get_numa_nodes() -> Result<Vec<usize>, std::io::Error> {
        // Implementation to discover NUMA topology
        Ok(vec![0, 1]) // Placeholder
    }
    
    fn get_current_numa_node() -> Result<usize, std::io::Error> {
        // Get current thread's NUMA node
        Ok(0) // Placeholder
    }
    
    fn set_memory_policy(node: usize) -> Result<(), std::io::Error> {
        // Set memory allocation policy for NUMA
        Ok(()) // Placeholder
    }
    
    fn get_cores_for_numa_node(node: usize) -> Result<Vec<usize>, std::io::Error> {
        // Get CPU cores for specific NUMA node
        Ok(vec![0, 1, 2, 3]) // Placeholder
    }
    
    fn distribute_threads_on_cores(cores: &[usize]) -> Result<(), std::io::Error> {
        // Distribute worker threads across cores
        Ok(()) // Placeholder
    }
}
```

## I/O Optimization

### Asynchronous I/O

```rust
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};
use futures::stream::{Stream, StreamExt};

pub struct AsyncModelIO;

impl AsyncModelIO {
    pub async fn save_model_async(
        &self,
        model: &NHITS,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let serialized = bincode::serialize(model)?;
        
        let file = File::create(path).await?;
        let mut writer = BufWriter::new(file);
        
        // Write in chunks to avoid blocking
        for chunk in serialized.chunks(8192) {
            writer.write_all(chunk).await?;
        }
        
        writer.flush().await?;
        Ok(())
    }
    
    pub async fn load_model_async(
        &self,
        path: &str,
    ) -> Result<NHITS, Box<dyn std::error::Error>> {
        let file = File::open(path).await?;
        let mut reader = BufReader::new(file);
        
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer).await?;
        
        let model: NHITS = bincode::deserialize(&buffer)?;
        Ok(model)
    }
    
    pub async fn stream_data_chunks<S>(
        &self,
        mut stream: S,
        chunk_size: usize,
    ) -> Result<Vec<Array3<f64>>, Box<dyn std::error::Error>>
    where
        S: Stream<Item = Result<Vec<f64>, Box<dyn std::error::Error>>> + Unpin,
    {
        let mut chunks = Vec::new();
        let mut current_chunk = Vec::new();
        
        while let Some(data_result) = stream.next().await {
            let data = data_result?;
            current_chunk.extend(data);
            
            if current_chunk.len() >= chunk_size {
                let chunk_array = self.vec_to_array3(current_chunk.drain(..chunk_size).collect())?;
                chunks.push(chunk_array);
            }
        }
        
        // Handle remaining data
        if !current_chunk.is_empty() {
            let chunk_array = self.vec_to_array3(current_chunk)?;
            chunks.push(chunk_array);
        }
        
        Ok(chunks)
    }
    
    fn vec_to_array3(&self, data: Vec<f64>) -> Result<Array3<f64>, Box<dyn std::error::Error>> {
        // Convert flat vector to 3D array based on expected dimensions
        let shape = (1, data.len(), 1); // Adjust based on your needs
        Ok(Array3::from_shape_vec(shape, data)?)
    }
}
```

### Database Optimization

```rust
use sqlx::{Pool, Postgres, Row};
use tokio_stream::StreamExt;

pub struct OptimizedDatabaseOps {
    pool: Pool<Postgres>,
}

impl OptimizedDatabaseOps {
    pub async fn batch_insert_predictions(
        &self,
        predictions: &[PredictionResult],
    ) -> Result<(), sqlx::Error> {
        // Use COPY for bulk inserts (much faster than individual INSERTs)
        let mut tx = self.pool.begin().await?;
        
        // Prepare data for COPY
        let mut copy_data = String::new();
        for pred in predictions {
            copy_data.push_str(&format!(
                "{}\t{}\t{}\t{}\n",
                pred.timestamp.timestamp(),
                pred.horizon,
                pred.value,
                pred.confidence
            ));
        }
        
        // Execute COPY command
        let query = "COPY predictions (timestamp, horizon, value, confidence) FROM STDIN";
        sqlx::raw_sql(query)
            .execute(&mut tx)
            .await?;
        
        tx.commit().await?;
        Ok(())
    }
    
    pub async fn stream_historical_data(
        &self,
        start_time: chrono::DateTime<chrono::Utc>,
        end_time: chrono::DateTime<chrono::Utc>,
    ) -> impl Stream<Item = Result<TimeSeriesPoint, sqlx::Error>> + '_ {
        sqlx::query_as!(
            TimeSeriesPoint,
            "SELECT timestamp, value FROM time_series WHERE timestamp BETWEEN $1 AND $2 ORDER BY timestamp",
            start_time,
            end_time
        )
        .fetch(&self.pool)
        .map(|result| result.map_err(sqlx::Error::from))
    }
    
    pub async fn get_model_metadata_cached(
        &self,
        model_id: &str,
        cache: &mut HashMap<String, ModelMetadata>,
    ) -> Result<ModelMetadata, sqlx::Error> {
        if let Some(metadata) = cache.get(model_id) {
            return Ok(metadata.clone());
        }
        
        let metadata = sqlx::query_as!(
            ModelMetadata,
            "SELECT * FROM model_metadata WHERE id = $1",
            model_id
        )
        .fetch_one(&self.pool)
        .await?;
        
        cache.insert(model_id.to_string(), metadata.clone());
        Ok(metadata)
    }
}
```

## Distributed Computing

### Model Sharding

```rust
use std::collections::HashMap;
use tokio::sync::mpsc;

pub struct ShardedNHITS {
    shards: HashMap<usize, NHITS>,
    coordinator: ModelCoordinator,
}

impl ShardedNHITS {
    pub fn new(num_shards: usize, base_config: NHITSConfig) -> Self {
        let mut shards = HashMap::new();
        
        for shard_id in 0..num_shards {
            let mut config = base_config.clone();
            // Adjust configuration per shard
            config.forecast_horizon = config.forecast_horizon / num_shards;
            
            let consciousness = Arc::new(ConsciousnessField::new());
            let autopoietic = Arc::new(AutopoieticSystem::new());
            let model = NHITS::new(config, consciousness, autopoietic);
            
            shards.insert(shard_id, model);
        }
        
        Self {
            shards,
            coordinator: ModelCoordinator::new(num_shards),
        }
    }
    
    pub async fn distributed_forecast(
        &mut self,
        input: &Array3<f64>,
        lookback: usize,
        total_horizon: usize,
    ) -> Result<Array3<f64>, NHITSError> {
        let shard_horizon = total_horizon / self.shards.len();
        let mut tasks = Vec::new();
        
        for (&shard_id, model) in &mut self.shards {
            let input_clone = input.clone();
            let mut model_clone = model.clone();
            
            let task = tokio::spawn(async move {
                model_clone.forward(&input_clone, lookback, shard_horizon)
            });
            
            tasks.push((shard_id, task));
        }
        
        // Collect results
        let mut shard_results = HashMap::new();
        for (shard_id, task) in tasks {
            let result = task.await.map_err(|e| {
                NHITSError::ComputationError(format!("Shard {} error: {}", shard_id, e))
            })??;
            shard_results.insert(shard_id, result);
        }
        
        // Combine shard results
        self.coordinator.combine_shard_results(shard_results)
    }
}

pub struct ModelCoordinator {
    num_shards: usize,
}

impl ModelCoordinator {
    pub fn new(num_shards: usize) -> Self {
        Self { num_shards }
    }
    
    pub fn combine_shard_results(
        &self,
        results: HashMap<usize, Array3<f64>>,
    ) -> Result<Array3<f64>, NHITSError> {
        // Concatenate results along time dimension
        let mut combined_results = Vec::new();
        
        for shard_id in 0..self.num_shards {
            if let Some(result) = results.get(&shard_id) {
                combined_results.push(result.clone());
            } else {
                return Err(NHITSError::ComputationError(
                    format!("Missing result from shard {}", shard_id)
                ));
            }
        }
        
        // Concatenate along horizon dimension
        concatenate_arrays_along_axis(&combined_results, 1)
    }
}
```

### Distributed Training

```rust
use tokio::sync::broadcast;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct DistributedTrainer {
    workers: Vec<TrainingWorker>,
    parameter_server: ParameterServer,
    coordinator: Arc<TrainingCoordinator>,
}

impl DistributedTrainer {
    pub async fn distributed_train(
        &mut self,
        train_data: &Array3<f64>,
        epochs: usize,
    ) -> Result<TrainingHistory, NHITSError> {
        let (tx, _rx) = broadcast::channel(1000);
        let coordinator = Arc::clone(&self.coordinator);
        
        // Split data across workers
        let data_splits = self.split_data(train_data)?;
        
        let mut worker_tasks = Vec::new();
        
        for (worker_id, (worker, data_split)) in 
            self.workers.iter_mut().zip(data_splits.iter()).enumerate() 
        {
            let tx_clone = tx.clone();
            let coordinator_clone = Arc::clone(&coordinator);
            let data_clone = data_split.clone();
            
            let task = tokio::spawn(async move {
                worker.train_worker(
                    worker_id,
                    &data_clone,
                    epochs,
                    tx_clone,
                    coordinator_clone,
                ).await
            });
            
            worker_tasks.push(task);
        }
        
        // Coordinate training
        let coordination_task = tokio::spawn(async move {
            coordinator.coordinate_training(epochs).await
        });
        
        // Wait for all workers to complete
        let mut worker_histories = Vec::new();
        for task in worker_tasks {
            let history = task.await.map_err(|e| {
                NHITSError::ComputationError(format!("Worker training error: {}", e))
            })??;
            worker_histories.push(history);
        }
        
        let _coordination_result = coordination_task.await.map_err(|e| {
            NHITSError::ComputationError(format!("Coordination error: {}", e))
        })??;
        
        // Aggregate training histories
        self.aggregate_training_histories(worker_histories)
    }
    
    fn split_data(&self, data: &Array3<f64>) -> Result<Vec<Array3<f64>>, NHITSError> {
        let num_workers = self.workers.len();
        let batch_size = data.shape()[0];
        let worker_batch_size = batch_size / num_workers;
        
        let mut splits = Vec::new();
        
        for i in 0..num_workers {
            let start_idx = i * worker_batch_size;
            let end_idx = if i == num_workers - 1 {
                batch_size // Last worker gets remaining data
            } else {
                (i + 1) * worker_batch_size
            };
            
            let split = data.slice(s![start_idx..end_idx, .., ..]).to_owned();
            splits.push(split);
        }
        
        Ok(splits)
    }
    
    fn aggregate_training_histories(
        &self,
        histories: Vec<TrainingHistory>,
    ) -> Result<TrainingHistory, NHITSError> {
        // Aggregate training metrics across workers
        let max_epochs = histories.iter().map(|h| h.train_losses.len()).max().unwrap_or(0);
        
        let mut aggregated = TrainingHistory {
            train_losses: Vec::with_capacity(max_epochs),
            val_losses: Vec::with_capacity(max_epochs),
            best_epoch: 0,
            best_val_loss: f64::INFINITY,
        };
        
        for epoch in 0..max_epochs {
            let mut epoch_train_losses = Vec::new();
            let mut epoch_val_losses = Vec::new();
            
            for history in &histories {
                if epoch < history.train_losses.len() {
                    epoch_train_losses.push(history.train_losses[epoch]);
                }
                if epoch < history.val_losses.len() {
                    epoch_val_losses.push(history.val_losses[epoch]);
                }
            }
            
            // Average losses across workers
            let avg_train_loss = epoch_train_losses.iter().sum::<f64>() / epoch_train_losses.len() as f64;
            let avg_val_loss = epoch_val_losses.iter().sum::<f64>() / epoch_val_losses.len() as f64;
            
            aggregated.train_losses.push(avg_train_loss);
            aggregated.val_losses.push(avg_val_loss);
            
            if avg_val_loss < aggregated.best_val_loss {
                aggregated.best_val_loss = avg_val_loss;
                aggregated.best_epoch = epoch;
            }
        }
        
        Ok(aggregated)
    }
}
```

## GPU Acceleration

### CUDA Integration

```rust
use cudarc::driver::{CudaDevice, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;

pub struct CudaNHITS {
    device: Arc<CudaDevice>,
    kernels: CudaKernels,
    streams: Vec<cudarc::driver::CudaStream>,
}

impl CudaNHITS {
    pub fn new(device_id: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let device = CudaDevice::new(device_id)?;
        let kernels = CudaKernels::load(&device)?;
        
        // Create multiple streams for concurrent execution
        let streams = (0..4)
            .map(|_| device.fork_default_stream())
            .collect::<Result<Vec<_>, _>>()?;
        
        Ok(Self {
            device: Arc::new(device),
            kernels,
            streams,
        })
    }
    
    pub async fn forward_cuda(
        &self,
        input: &Array3<f64>,
        lookback: usize,
        horizon: usize,
    ) -> Result<Array3<f64>, Box<dyn std::error::Error>> {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let seq_len = input_shape[1];
        let features = input_shape[2];
        
        // Allocate GPU memory
        let input_gpu = self.device.htod_copy(input.as_slice().unwrap())?;
        let output_gpu = self.device.alloc_zeros::<f64>(batch_size * horizon * features)?;
        
        // Launch CUDA kernels
        let cfg = LaunchConfig::for_num_elems((batch_size * seq_len * features) as u32);
        
        // Hierarchical processing kernel
        unsafe {
            self.kernels.hierarchical_forward.launch(
                cfg,
                (
                    &input_gpu,
                    &output_gpu,
                    batch_size as i32,
                    seq_len as i32,
                    features as i32,
                    horizon as i32,
                ),
            )?;
        }
        
        // Synchronize and copy result back
        self.device.synchronize()?;
        let output_host = self.device.dtoh_sync_copy(&output_gpu)?;
        
        // Reshape to output array
        let output_array = Array3::from_shape_vec(
            (batch_size, horizon, features),
            output_host,
        )?;
        
        Ok(output_array)
    }
    
    pub async fn parallel_ensemble_cuda(
        &self,
        models: &[CudaNHITS],
        input: &Array3<f64>,
        lookback: usize,
        horizon: usize,
    ) -> Result<Array3<f64>, Box<dyn std::error::Error>> {
        let mut tasks = Vec::new();
        
        for (i, model) in models.iter().enumerate() {
            let input_clone = input.clone();
            let stream_id = i % self.streams.len();
            
            let task = tokio::spawn(async move {
                model.forward_cuda_stream(&input_clone, lookback, horizon, stream_id).await
            });
            
            tasks.push(task);
        }
        
        // Collect results
        let mut results = Vec::new();
        for task in tasks {
            let result = task.await??;
            results.push(result);
        }
        
        // Average ensemble predictions on GPU
        self.average_predictions_cuda(&results).await
    }
    
    async fn average_predictions_cuda(
        &self,
        predictions: &[Array3<f64>],
    ) -> Result<Array3<f64>, Box<dyn std::error::Error>> {
        if predictions.is_empty() {
            return Err("No predictions to average".into());
        }
        
        let shape = predictions[0].shape();
        let total_elements = shape.iter().product::<usize>();
        
        // Allocate GPU memory for averaging
        let mut gpu_arrays = Vec::new();
        for pred in predictions {
            let gpu_array = self.device.htod_copy(pred.as_slice().unwrap())?;
            gpu_arrays.push(gpu_array);
        }
        
        let result_gpu = self.device.alloc_zeros::<f64>(total_elements)?;
        
        // Launch averaging kernel
        let cfg = LaunchConfig::for_num_elems(total_elements as u32);
        unsafe {
            self.kernels.average_arrays.launch(
                cfg,
                (
                    &gpu_arrays,
                    &result_gpu,
                    total_elements as i32,
                    predictions.len() as i32,
                ),
            )?;
        }
        
        // Copy result back
        self.device.synchronize()?;
        let result_host = self.device.dtoh_sync_copy(&result_gpu)?;
        
        Ok(Array3::from_shape_vec(shape.to_vec(), result_host)?)
    }
}

struct CudaKernels {
    hierarchical_forward: cudarc::driver::CudaFunction,
    attention_kernel: cudarc::driver::CudaFunction,
    average_arrays: cudarc::driver::CudaFunction,
}

impl CudaKernels {
    fn load(device: &CudaDevice) -> Result<Self, Box<dyn std::error::Error>> {
        let ptx = Ptx::from_src(include_str!("nhits_kernels.cu"));
        device.load_ptx(ptx, "nhits_kernels", &[
            "hierarchical_forward",
            "attention_kernel", 
            "average_arrays"
        ])?;
        
        Ok(Self {
            hierarchical_forward: device.get_func("nhits_kernels", "hierarchical_forward")?,
            attention_kernel: device.get_func("nhits_kernels", "attention_kernel")?,
            average_arrays: device.get_func("nhits_kernels", "average_arrays")?,
        })
    }
}
```

## Caching Strategies

### Multi-Level Caching

```rust
use std::sync::Arc;
use moka::future::Cache;
use std::hash::{Hash, Hasher};

pub struct NHITSCache {
    l1_cache: Arc<Cache<CacheKey, Array3<f64>>>,      // In-memory cache
    l2_cache: Arc<Cache<CacheKey, Vec<u8>>>,          // Compressed cache
    prediction_cache: Arc<Cache<PredictionKey, ForecastResult>>,
    model_cache: Arc<Cache<String, NHITS>>,
}

#[derive(Hash, Clone, Eq, PartialEq)]
struct CacheKey {
    input_hash: u64,
    lookback: usize,
    horizon: usize,
    model_version: String,
}

#[derive(Hash, Clone, Eq, PartialEq)]
struct PredictionKey {
    timestamp: i64,
    horizon: usize,
    features_hash: u64,
}

impl NHITSCache {
    pub fn new() -> Self {
        Self {
            l1_cache: Arc::new(
                Cache::builder()
                    .max_capacity(1000)
                    .time_to_live(Duration::from_secs(300)) // 5 minutes
                    .build(),
            ),
            l2_cache: Arc::new(
                Cache::builder()
                    .max_capacity(5000)
                    .time_to_live(Duration::from_secs(3600)) // 1 hour
                    .build(),
            ),
            prediction_cache: Arc::new(
                Cache::builder()
                    .max_capacity(10000)
                    .time_to_live(Duration::from_secs(1800)) // 30 minutes
                    .build(),
            ),
            model_cache: Arc::new(
                Cache::builder()
                    .max_capacity(10)
                    .time_to_live(Duration::from_secs(7200)) // 2 hours
                    .build(),
            ),
        }
    }
    
    pub async fn get_or_compute_forecast(
        &self,
        input: &Array3<f64>,
        model: &NHITS,
        lookback: usize,
        horizon: usize,
    ) -> Result<Array3<f64>, NHITSError> {
        let cache_key = CacheKey {
            input_hash: self.hash_array(input),
            lookback,
            horizon,
            model_version: model.get_version(),
        };
        
        // Try L1 cache first
        if let Some(cached_result) = self.l1_cache.get(&cache_key).await {
            return Ok(cached_result);
        }
        
        // Try L2 cache (compressed)
        if let Some(compressed_data) = self.l2_cache.get(&cache_key).await {
            if let Ok(decompressed) = self.decompress_array(&compressed_data) {
                // Store in L1 cache for faster access
                self.l1_cache.insert(cache_key.clone(), decompressed.clone()).await;
                return Ok(decompressed);
            }
        }
        
        // Compute new prediction
        let mut model_clone = model.clone();
        let result = model_clone.forward(input, lookback, horizon)?;
        
        // Store in both caches
        self.l1_cache.insert(cache_key.clone(), result.clone()).await;
        
        if let Ok(compressed) = self.compress_array(&result) {
            self.l2_cache.insert(cache_key, compressed).await;
        }
        
        Ok(result)
    }
    
    pub async fn cache_prediction(
        &self,
        timestamp: chrono::DateTime<chrono::Utc>,
        horizon: usize,
        features: &Array1<f64>,
        result: ForecastResult,
    ) {
        let key = PredictionKey {
            timestamp: timestamp.timestamp(),
            horizon,
            features_hash: self.hash_array1d(features),
        };
        
        self.prediction_cache.insert(key, result).await;
    }
    
    pub async fn get_cached_prediction(
        &self,
        timestamp: chrono::DateTime<chrono::Utc>,
        horizon: usize,
        features: &Array1<f64>,
    ) -> Option<ForecastResult> {
        let key = PredictionKey {
            timestamp: timestamp.timestamp(),
            horizon,
            features_hash: self.hash_array1d(features),
        };
        
        self.prediction_cache.get(&key).await
    }
    
    fn hash_array(&self, array: &Array3<f64>) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        array.shape().hash(&mut hasher);
        
        // Hash a sample of values for performance
        let sample_size = (array.len() / 100).max(1);
        for (i, &value) in array.iter().enumerate() {
            if i % sample_size == 0 {
                value.to_bits().hash(&mut hasher);
            }
        }
        
        hasher.finish()
    }
    
    fn hash_array1d(&self, array: &Array1<f64>) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for &value in array {
            value.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }
    
    fn compress_array(&self, array: &Array3<f64>) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let serialized = bincode::serialize(array)?;
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&serialized)?;
        Ok(encoder.finish()?)
    }
    
    fn decompress_array(&self, data: &[u8]) -> Result<Array3<f64>, Box<dyn std::error::Error>> {
        let mut decoder = flate2::read::GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        let array: Array3<f64> = bincode::deserialize(&decompressed)?;
        Ok(array)
    }
}
```

## Monitoring & Profiling

### Performance Profiler

```rust
use std::time::{Duration, Instant};
use std::collections::HashMap;
use prometheus::{Counter, Histogram, Gauge, Registry};

pub struct NHITSProfiler {
    timers: HashMap<String, Instant>,
    counters: HashMap<String, u64>,
    durations: HashMap<String, Vec<Duration>>,
    
    // Prometheus metrics
    forward_duration: Histogram,
    training_duration: Histogram,
    memory_usage: Gauge,
    cache_hits: Counter,
    cache_misses: Counter,
}

impl NHITSProfiler {
    pub fn new(registry: &Registry) -> Self {
        let forward_duration = Histogram::new(
            "nhits_forward_duration_seconds",
            "Time spent in forward pass",
        ).unwrap();
        
        let training_duration = Histogram::new(
            "nhits_training_duration_seconds", 
            "Time spent in training",
        ).unwrap();
        
        let memory_usage = Gauge::new(
            "nhits_memory_usage_bytes",
            "Current memory usage",
        ).unwrap();
        
        let cache_hits = Counter::new(
            "nhits_cache_hits_total",
            "Number of cache hits",
        ).unwrap();
        
        let cache_misses = Counter::new(
            "nhits_cache_misses_total",
            "Number of cache misses",
        ).unwrap();
        
        registry.register(Box::new(forward_duration.clone())).unwrap();
        registry.register(Box::new(training_duration.clone())).unwrap();
        registry.register(Box::new(memory_usage.clone())).unwrap();
        registry.register(Box::new(cache_hits.clone())).unwrap();
        registry.register(Box::new(cache_misses.clone())).unwrap();
        
        Self {
            timers: HashMap::new(),
            counters: HashMap::new(),
            durations: HashMap::new(),
            forward_duration,
            training_duration,
            memory_usage,
            cache_hits,
            cache_misses,
        }
    }
    
    pub fn start_timer(&mut self, name: &str) {
        self.timers.insert(name.to_string(), Instant::now());
    }
    
    pub fn end_timer(&mut self, name: &str) -> Option<Duration> {
        if let Some(start) = self.timers.remove(name) {
            let duration = start.elapsed();
            
            // Record duration
            self.durations.entry(name.to_string())
                .or_insert_with(Vec::new)
                .push(duration);
            
            // Update Prometheus metrics
            match name {
                "forward_pass" => self.forward_duration.observe(duration.as_secs_f64()),
                "training" => self.training_duration.observe(duration.as_secs_f64()),
                _ => {}
            }
            
            Some(duration)
        } else {
            None
        }
    }
    
    pub fn increment_counter(&mut self, name: &str) {
        *self.counters.entry(name.to_string()).or_insert(0) += 1;
        
        match name {
            "cache_hit" => self.cache_hits.inc(),
            "cache_miss" => self.cache_misses.inc(),
            _ => {}
        }
    }
    
    pub fn update_memory_usage(&self, bytes: u64) {
        self.memory_usage.set(bytes as f64);
    }
    
    pub fn get_stats(&self) -> ProfilerStats {
        let mut avg_durations = HashMap::new();
        let mut max_durations = HashMap::new();
        let mut min_durations = HashMap::new();
        
        for (name, durations) in &self.durations {
            if !durations.is_empty() {
                let avg = durations.iter().sum::<Duration>() / durations.len() as u32;
                let max = *durations.iter().max().unwrap();
                let min = *durations.iter().min().unwrap();
                
                avg_durations.insert(name.clone(), avg);
                max_durations.insert(name.clone(), max);
                min_durations.insert(name.clone(), min);
            }
        }
        
        ProfilerStats {
            counters: self.counters.clone(),
            avg_durations,
            max_durations,
            min_durations,
        }
    }
    
    pub fn reset(&mut self) {
        self.timers.clear();
        self.counters.clear();
        self.durations.clear();
    }
}

#[derive(Debug, Clone)]
pub struct ProfilerStats {
    pub counters: HashMap<String, u64>,
    pub avg_durations: HashMap<String, Duration>,
    pub max_durations: HashMap<String, Duration>,
    pub min_durations: HashMap<String, Duration>,
}
```

## Benchmarking

### Performance Benchmark Suite

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::sync::Arc;

pub fn benchmark_nhits_performance(c: &mut Criterion) {
    let consciousness = Arc::new(ConsciousnessField::new());
    let autopoietic = Arc::new(AutopoieticSystem::new());
    
    let mut group = c.benchmark_group("nhits_forward_pass");
    
    // Benchmark different input sizes
    for &input_size in &[32, 64, 128, 256, 512] {
        let config = NHITSConfigBuilder::new()
            .with_lookback(168)
            .with_horizon(24)
            .with_features(1, 1)
            .build()
            .unwrap();
        
        let mut model = NHITS::new(config, consciousness.clone(), autopoietic.clone());
        let input = Array3::zeros((input_size, 168, 1));
        
        group.bench_with_input(
            BenchmarkId::new("forward_pass", input_size),
            &input_size,
            |b, &_size| {
                b.iter(|| {
                    let result = model.forward(black_box(&input), 168, 24);
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
    
    // Benchmark different configurations
    let mut config_group = c.benchmark_group("nhits_configurations");
    
    let test_configs = vec![
        ("minimal", NHITSConfig::minimal()),
        ("default", NHITSConfig::default()),
        ("high_accuracy", NHITSConfig::for_use_case(UseCase::LongTermForecasting)),
        ("high_frequency", NHITSConfig::for_use_case(UseCase::HighFrequencyTrading)),
    ];
    
    for (config_name, config) in test_configs {
        let mut model = NHITS::new(config, consciousness.clone(), autopoietic.clone());
        let input = Array3::zeros((32, 168, 1));
        
        config_group.bench_function(config_name, |b| {
            b.iter(|| {
                let result = model.forward(black_box(&input), 168, 24);
                black_box(result)
            });
        });
    }
    
    config_group.finish();
    
    // Benchmark training performance
    let mut training_group = c.benchmark_group("nhits_training");
    
    let config = NHITSConfig::minimal();
    let mut model = NHITS::new(config, consciousness.clone(), autopoietic.clone());
    let train_data = Array3::zeros((32, 168, 1));
    
    training_group.bench_function("single_epoch", |b| {
        b.iter(|| {
            let result = model.train(black_box(&train_data), None, 1);
            black_box(result)
        });
    });
    
    training_group.finish();
}

pub fn benchmark_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    
    // Benchmark memory allocation patterns
    group.bench_function("array_allocation", |b| {
        b.iter(|| {
            let array = Array3::<f64>::zeros(black_box((1000, 168, 10)));
            black_box(array)
        });
    });
    
    group.bench_function("model_creation", |b| {
        b.iter(|| {
            let consciousness = Arc::new(ConsciousnessField::new());
            let autopoietic = Arc::new(AutopoieticSystem::new());
            let config = NHITSConfig::minimal();
            let model = NHITS::new(config, consciousness, autopoietic);
            black_box(model)
        });
    });
    
    group.finish();
}

pub fn benchmark_cache_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_performance");
    
    let cache = NHITSCache::new();
    let test_data = Array3::zeros((32, 168, 1));
    
    group.bench_function("cache_miss", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        b.to_async(rt).iter(|| async {
            let key = format!("test_key_{}", rand::random::<u64>());
            let result = cache.get_compressed(&key).await;
            black_box(result)
        });
    });
    
    // Pre-populate cache for hit benchmark
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        cache.store_compressed("cached_key", &test_data).await.unwrap();
    });
    
    group.bench_function("cache_hit", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        b.to_async(rt).iter(|| async {
            let result = cache.get_compressed("cached_key").await;
            black_box(result)
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_nhits_performance,
    benchmark_memory_usage,
    benchmark_cache_performance
);
criterion_main!(benches);
```

### Performance Testing Framework

```rust
pub struct PerformanceTestSuite;

impl PerformanceTestSuite {
    pub async fn run_comprehensive_tests() -> PerformanceReport {
        let mut report = PerformanceReport::new();
        
        // Test different workload patterns
        report.add_test_result("throughput", Self::test_throughput().await);
        report.add_test_result("latency", Self::test_latency().await);
        report.add_test_result("memory_efficiency", Self::test_memory_efficiency().await);
        report.add_test_result("scalability", Self::test_scalability().await);
        report.add_test_result("consciousness_overhead", Self::test_consciousness_overhead().await);
        
        report
    }
    
    async fn test_throughput() -> TestResult {
        let consciousness = Arc::new(ConsciousnessField::new());
        let autopoietic = Arc::new(AutopoieticSystem::new());
        let config = NHITSConfig::for_use_case(UseCase::HighFrequencyTrading);
        let mut model = NHITS::new(config, consciousness, autopoietic);
        
        let num_samples = 10000;
        let input = Array3::zeros((1, 50, 1));
        
        let start = Instant::now();
        for _ in 0..num_samples {
            let _ = model.forward(&input, 50, 1).unwrap();
        }
        let duration = start.elapsed();
        
        let throughput = num_samples as f64 / duration.as_secs_f64();
        
        TestResult {
            metric: "predictions_per_second".to_string(),
            value: throughput,
            target: 50000.0,
            passed: throughput >= 50000.0,
        }
    }
    
    async fn test_latency() -> TestResult {
        let consciousness = Arc::new(ConsciousnessField::new());
        let autopoietic = Arc::new(AutopoieticSystem::new());
        let config = NHITSConfig::minimal();
        let mut model = NHITS::new(config, consciousness, autopoietic);
        
        let input = Array3::zeros((1, 100, 1));
        let num_iterations = 1000;
        let mut latencies = Vec::new();
        
        for _ in 0..num_iterations {
            let start = Instant::now();
            let _ = model.forward(&input, 100, 10).unwrap();
            latencies.push(start.elapsed());
        }
        
        latencies.sort();
        let p95_latency = latencies[(num_iterations * 95 / 100) as usize];
        
        TestResult {
            metric: "p95_latency_ms".to_string(),
            value: p95_latency.as_millis() as f64,
            target: 50.0,
            passed: p95_latency.as_millis() <= 50,
        }
    }
    
    async fn test_memory_efficiency() -> TestResult {
        let initial_memory = Self::get_memory_usage();
        
        let consciousness = Arc::new(ConsciousnessField::new());
        let autopoietic = Arc::new(AutopoieticSystem::new());
        let config = NHITSConfig::default();
        let mut model = NHITS::new(config, consciousness, autopoietic);
        
        // Process large dataset
        let large_input = Array3::zeros((1000, 1000, 1));
        let _ = model.forward(&large_input, 1000, 100).unwrap();
        
        let peak_memory = Self::get_memory_usage();
        let memory_increase = peak_memory - initial_memory;
        
        TestResult {
            metric: "memory_increase_mb".to_string(),
            value: memory_increase as f64 / 1024.0 / 1024.0,
            target: 1000.0, // 1GB target
            passed: memory_increase < 1024 * 1024 * 1024, // Less than 1GB
        }
    }
    
    async fn test_scalability() -> TestResult {
        let base_latency = Self::measure_latency_for_batch_size(1).await;
        let scaled_latency = Self::measure_latency_for_batch_size(100).await;
        
        let scaling_factor = scaled_latency / base_latency;
        
        TestResult {
            metric: "scaling_factor".to_string(),
            value: scaling_factor,
            target: 50.0, // Should not be more than 50x slower for 100x more data
            passed: scaling_factor <= 50.0,
        }
    }
    
    async fn test_consciousness_overhead() -> TestResult {
        // Test with consciousness disabled
        let config_no_consciousness = NHITSConfigBuilder::new()
            .with_consciousness(false, 0.0)
            .build()
            .unwrap();
        
        let latency_no_consciousness = Self::measure_latency_with_config(config_no_consciousness).await;
        
        // Test with consciousness enabled
        let config_with_consciousness = NHITSConfigBuilder::new()
            .with_consciousness(true, 0.1)
            .build()
            .unwrap();
        
        let latency_with_consciousness = Self::measure_latency_with_config(config_with_consciousness).await;
        
        let overhead = (latency_with_consciousness / latency_no_consciousness - 1.0) * 100.0;
        
        TestResult {
            metric: "consciousness_overhead_percent".to_string(),
            value: overhead,
            target: 20.0, // Should not add more than 20% overhead
            passed: overhead <= 20.0,
        }
    }
    
    fn get_memory_usage() -> usize {
        // Platform-specific memory usage measurement
        #[cfg(target_os = "linux")]
        {
            let mut system = sysinfo::System::new();
            system.refresh_memory();
            system.used_memory() as usize
        }
        #[cfg(not(target_os = "linux"))]
        {
            0 // Placeholder for other platforms
        }
    }
    
    async fn measure_latency_for_batch_size(batch_size: usize) -> f64 {
        let consciousness = Arc::new(ConsciousnessField::new());
        let autopoietic = Arc::new(AutopoieticSystem::new());
        let config = NHITSConfig::minimal();
        let mut model = NHITS::new(config, consciousness, autopoietic);
        
        let input = Array3::zeros((batch_size, 100, 1));
        
        let start = Instant::now();
        let _ = model.forward(&input, 100, 10).unwrap();
        start.elapsed().as_secs_f64()
    }
    
    async fn measure_latency_with_config(config: NHITSConfig) -> f64 {
        let consciousness = Arc::new(ConsciousnessField::new());
        let autopoietic = Arc::new(AutopoieticSystem::new());
        let mut model = NHITS::new(config, consciousness, autopoietic);
        
        let input = Array3::zeros((32, 100, 1));
        let num_iterations = 100;
        
        let start = Instant::now();
        for _ in 0..num_iterations {
            let _ = model.forward(&input, 100, 10).unwrap();
        }
        start.elapsed().as_secs_f64() / num_iterations as f64
    }
}

#[derive(Debug)]
pub struct PerformanceReport {
    test_results: HashMap<String, TestResult>,
    overall_score: f64,
}

impl PerformanceReport {
    pub fn new() -> Self {
        Self {
            test_results: HashMap::new(),
            overall_score: 0.0,
        }
    }
    
    pub fn add_test_result(&mut self, test_name: &str, result: TestResult) {
        self.test_results.insert(test_name.to_string(), result);
        self.calculate_overall_score();
    }
    
    fn calculate_overall_score(&mut self) {
        let passed_tests = self.test_results.values().filter(|r| r.passed).count();
        let total_tests = self.test_results.len();
        
        self.overall_score = if total_tests > 0 {
            (passed_tests as f64 / total_tests as f64) * 100.0
        } else {
            0.0
        };
    }
    
    pub fn print_report(&self) {
        println!("NHITS Performance Test Report");
        println!("============================");
        println!("Overall Score: {:.1}%", self.overall_score);
        println!();
        
        for (test_name, result) in &self.test_results {
            let status = if result.passed { "PASS" } else { "FAIL" };
            println!("{}: {} - {}: {:.2} (target: {:.2})", 
                test_name, status, result.metric, result.value, result.target);
        }
    }
}

#[derive(Debug, Clone)]
pub struct TestResult {
    pub metric: String,
    pub value: f64,
    pub target: f64,
    pub passed: bool,
}
```

This comprehensive performance tuning guide provides detailed strategies and implementations for optimizing NHITS across all performance dimensions. Use these techniques based on your specific requirements and constraints.