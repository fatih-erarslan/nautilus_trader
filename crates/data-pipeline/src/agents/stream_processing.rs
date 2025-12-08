//! # Stream Processing Agent
//!
//! High-throughput stream processing agent with sub-100μs latency targets.
//! Provides real-time stream processing with adaptive buffering and SIMD optimizations.

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::{HashMap, VecDeque};
use tokio::sync::{RwLock, mpsc, Mutex};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error, debug};
use async_trait::async_trait;
use rayon::prelude::*;
use futures::stream::{Stream, StreamExt};

use crate::agents::base::{
    DataAgent, DataAgentId, DataAgentType, DataAgentState, DataAgentInfo,
    DataMessage, DataMessageType, MessageMetadata, MessagePriority,
    CoordinationMessage, HealthStatus, HealthLevel, HealthMetrics,
    AgentMetrics, BaseDataAgent, MetricsUpdate
};

/// Stream processing agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamProcessingConfig {
    /// Target processing latency in microseconds
    pub target_latency_us: u64,
    /// Buffer configuration
    pub buffer_config: BufferConfig,
    /// Processing configuration
    pub processing_config: ProcessingConfig,
    /// Parallelization settings
    pub parallel_config: ParallelConfig,
    /// SIMD optimization settings
    pub simd_config: SimdConfig,
    /// Memory optimization settings
    pub memory_config: MemoryConfig,
}

impl Default for StreamProcessingConfig {
    fn default() -> Self {
        Self {
            target_latency_us: 100,
            buffer_config: BufferConfig::default(),
            processing_config: ProcessingConfig::default(),
            parallel_config: ParallelConfig::default(),
            simd_config: SimdConfig::default(),
            memory_config: MemoryConfig::default(),
        }
    }
}

/// Buffer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferConfig {
    /// Buffer size for each stream
    pub buffer_size: usize,
    /// Maximum buffer size before dropping messages
    pub max_buffer_size: usize,
    /// Buffer strategy
    pub buffer_strategy: BufferStrategy,
    /// Adaptive buffering settings
    pub adaptive_buffering: AdaptiveBufferingConfig,
    /// Prefetch settings
    pub prefetch_config: PrefetchConfig,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            buffer_size: 100000,
            max_buffer_size: 1000000,
            buffer_strategy: BufferStrategy::RingBuffer,
            adaptive_buffering: AdaptiveBufferingConfig::default(),
            prefetch_config: PrefetchConfig::default(),
        }
    }
}

/// Buffer strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BufferStrategy {
    FIFO,
    LIFO,
    Priority,
    RingBuffer,
    Adaptive,
}

/// Adaptive buffering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveBufferingConfig {
    /// Enable adaptive buffering
    pub enabled: bool,
    /// Minimum buffer size
    pub min_buffer_size: usize,
    /// Maximum buffer size
    pub max_buffer_size: usize,
    /// Adaptation algorithm
    pub adaptation_algorithm: AdaptationAlgorithm,
    /// Adaptation interval in milliseconds
    pub adaptation_interval_ms: u64,
}

impl Default for AdaptiveBufferingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_buffer_size: 1000,
            max_buffer_size: 1000000,
            adaptation_algorithm: AdaptationAlgorithm::PID,
            adaptation_interval_ms: 100,
        }
    }
}

/// Adaptation algorithms
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AdaptationAlgorithm {
    Simple,
    PID,
    Reinforcement,
    Neural,
}

/// Prefetch configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefetchConfig {
    /// Enable prefetching
    pub enabled: bool,
    /// Prefetch distance
    pub distance: usize,
    /// Prefetch strategy
    pub strategy: PrefetchStrategy,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            distance: 4,
            strategy: PrefetchStrategy::Sequential,
        }
    }
}

/// Prefetch strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PrefetchStrategy {
    Sequential,
    Strided,
    Random,
}

/// Processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Batch size for processing
    pub batch_size: usize,
    /// Processing mode
    pub processing_mode: ProcessingMode,
    /// Window configuration
    pub window_config: WindowConfig,
    /// Aggregation settings
    pub aggregation_config: AggregationConfig,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            processing_mode: ProcessingMode::Streaming,
            window_config: WindowConfig::default(),
            aggregation_config: AggregationConfig::default(),
        }
    }
}

/// Processing modes
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ProcessingMode {
    Streaming,
    Batch,
    MicroBatch,
    Hybrid,
}

/// Window configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowConfig {
    /// Window type
    pub window_type: WindowType,
    /// Window size
    pub window_size: Duration,
    /// Slide interval
    pub slide_interval: Duration,
    /// Watermark delay
    pub watermark_delay: Duration,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            window_type: WindowType::Tumbling,
            window_size: Duration::from_millis(1000),
            slide_interval: Duration::from_millis(100),
            watermark_delay: Duration::from_millis(100),
        }
    }
}

/// Window types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WindowType {
    Tumbling,
    Sliding,
    Session,
    Global,
}

/// Aggregation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationConfig {
    /// Aggregation functions
    pub functions: Vec<AggregationFunction>,
    /// Group by fields
    pub group_by: Vec<String>,
    /// Output format
    pub output_format: OutputFormat,
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            functions: vec![
                AggregationFunction::Count,
                AggregationFunction::Sum,
                AggregationFunction::Average,
                AggregationFunction::Min,
                AggregationFunction::Max,
            ],
            group_by: Vec::new(),
            output_format: OutputFormat::JSON,
        }
    }
}

/// Aggregation functions
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AggregationFunction {
    Count,
    Sum,
    Average,
    Min,
    Max,
    Median,
    Variance,
    StandardDeviation,
}

/// Output formats
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OutputFormat {
    JSON,
    Binary,
    Protobuf,
    Avro,
}

/// Parallelization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Number of worker threads
    pub worker_threads: usize,
    /// Thread affinity
    pub thread_affinity: Vec<usize>,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Work stealing
    pub work_stealing: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            worker_threads: num_cpus::get(),
            thread_affinity: (0..num_cpus::get()).collect(),
            load_balancing: LoadBalancingStrategy::RoundRobin,
            work_stealing: true,
        }
    }
}

/// Load balancing strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    Random,
    Hash,
    Adaptive,
}

/// SIMD configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimdConfig {
    /// Enable SIMD operations
    pub enabled: bool,
    /// SIMD instruction set
    pub instruction_set: SimdInstructionSet,
    /// Vector width
    pub vector_width: usize,
    /// Optimization level
    pub optimization_level: SimdOptimizationLevel,
}

impl Default for SimdConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            instruction_set: SimdInstructionSet::AVX2,
            vector_width: 8,
            optimization_level: SimdOptimizationLevel::Aggressive,
        }
    }
}

/// SIMD instruction sets
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SimdInstructionSet {
    SSE,
    SSE2,
    AVX,
    AVX2,
    AVX512,
}

/// SIMD optimization levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SimdOptimizationLevel {
    Conservative,
    Moderate,
    Aggressive,
}

/// Memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Memory pool size in MB
    pub pool_size_mb: usize,
    /// Memory alignment
    pub alignment: usize,
    /// NUMA awareness
    pub numa_aware: bool,
    /// Huge pages
    pub huge_pages: bool,
    /// Memory prefetch
    pub memory_prefetch: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            pool_size_mb: 1024,
            alignment: 64, // Cache line aligned
            numa_aware: true,
            huge_pages: true,
            memory_prefetch: true,
        }
    }
}

/// Stream buffer with adaptive capabilities
pub struct AdaptiveBuffer<T> {
    buffer: VecDeque<T>,
    max_size: usize,
    current_size: usize,
    adaptation_config: AdaptiveBufferingConfig,
    last_adaptation: Instant,
    pid_controller: PIDController,
}

/// PID Controller for adaptive buffering
#[derive(Debug, Clone)]
pub struct PIDController {
    kp: f64,
    ki: f64,
    kd: f64,
    previous_error: f64,
    integral: f64,
    setpoint: f64,
}

impl PIDController {
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self {
            kp,
            ki,
            kd,
            previous_error: 0.0,
            integral: 0.0,
            setpoint: 0.0,
        }
    }
    
    pub fn update(&mut self, measurement: f64, dt: f64) -> f64 {
        let error = self.setpoint - measurement;
        self.integral += error * dt;
        let derivative = (error - self.previous_error) / dt;
        
        let output = self.kp * error + self.ki * self.integral + self.kd * derivative;
        
        self.previous_error = error;
        output
    }
}

impl<T> AdaptiveBuffer<T> {
    pub fn new(config: AdaptiveBufferingConfig) -> Self {
        Self {
            buffer: VecDeque::with_capacity(config.min_buffer_size),
            max_size: config.max_buffer_size,
            current_size: config.min_buffer_size,
            adaptation_config: config,
            last_adaptation: Instant::now(),
            pid_controller: PIDController::new(1.0, 0.1, 0.01),
        }
    }
    
    pub fn push(&mut self, item: T) -> Result<()> {
        if self.buffer.len() >= self.current_size {
            if self.buffer.len() >= self.max_size {
                // Drop oldest item
                self.buffer.pop_front();
            }
        }
        
        self.buffer.push_back(item);
        self.adapt_if_needed();
        
        Ok(())
    }
    
    pub fn pop(&mut self) -> Option<T> {
        let item = self.buffer.pop_front();
        self.adapt_if_needed();
        item
    }
    
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
    
    fn adapt_if_needed(&mut self) {
        if !self.adaptation_config.enabled {
            return;
        }
        
        let now = Instant::now();
        if now.duration_since(self.last_adaptation).as_millis() >= self.adaptation_config.adaptation_interval_ms as u128 {
            self.adapt_buffer_size();
            self.last_adaptation = now;
        }
    }
    
    fn adapt_buffer_size(&mut self) {
        let utilization = self.buffer.len() as f64 / self.current_size as f64;
        
        match self.adaptation_config.adaptation_algorithm {
            AdaptationAlgorithm::Simple => {
                if utilization > 0.8 {
                    self.current_size = (self.current_size * 2).min(self.max_size);
                } else if utilization < 0.2 {
                    self.current_size = (self.current_size / 2).max(self.adaptation_config.min_buffer_size);
                }
            }
            AdaptationAlgorithm::PID => {
                self.pid_controller.setpoint = 0.5; // Target 50% utilization
                let adjustment = self.pid_controller.update(utilization, 0.1);
                
                let new_size = self.current_size as f64 + adjustment * self.current_size as f64;
                self.current_size = (new_size as usize)
                    .max(self.adaptation_config.min_buffer_size)
                    .min(self.max_size);
            }
            _ => {
                // Other algorithms would be implemented here
            }
        }
    }
}

/// Stream data item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamDataItem {
    pub id: uuid::Uuid,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub source: String,
    pub data: serde_json::Value,
    pub metadata: StreamMetadata,
}

/// Stream metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMetadata {
    pub stream_id: String,
    pub sequence_number: u64,
    pub partition_key: String,
    pub priority: MessagePriority,
    pub watermark: chrono::DateTime<chrono::Utc>,
}

/// Stream processing agent
pub struct StreamProcessingAgent {
    base: BaseDataAgent,
    config: Arc<StreamProcessingConfig>,
    buffers: Arc<RwLock<HashMap<String, AdaptiveBuffer<StreamDataItem>>>>,
    processing_metrics: Arc<RwLock<StreamProcessingMetrics>>,
    state: Arc<RwLock<StreamProcessingState>>,
    worker_pool: Arc<rayon::ThreadPool>,
}

/// Stream processing metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamProcessingMetrics {
    pub messages_processed: u64,
    pub messages_buffered: u64,
    pub messages_dropped: u64,
    pub average_processing_time_us: f64,
    pub max_processing_time_us: f64,
    pub throughput_ops_per_sec: f64,
    pub buffer_utilization: f64,
    pub worker_utilization: f64,
    pub last_update: chrono::DateTime<chrono::Utc>,
}

impl Default for StreamProcessingMetrics {
    fn default() -> Self {
        Self {
            messages_processed: 0,
            messages_buffered: 0,
            messages_dropped: 0,
            average_processing_time_us: 0.0,
            max_processing_time_us: 0.0,
            throughput_ops_per_sec: 0.0,
            buffer_utilization: 0.0,
            worker_utilization: 0.0,
            last_update: chrono::Utc::now(),
        }
    }
}

/// Stream processing state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamProcessingState {
    pub active_streams: usize,
    pub total_buffer_size: usize,
    pub active_workers: usize,
    pub is_healthy: bool,
    pub last_health_check: chrono::DateTime<chrono::Utc>,
}

impl Default for StreamProcessingState {
    fn default() -> Self {
        Self {
            active_streams: 0,
            total_buffer_size: 0,
            active_workers: 0,
            is_healthy: true,
            last_health_check: chrono::Utc::now(),
        }
    }
}

impl StreamProcessingAgent {
    /// Create a new stream processing agent
    pub async fn new(config: StreamProcessingConfig) -> Result<Self> {
        let base = BaseDataAgent::new(DataAgentType::StreamProcessing);
        let config = Arc::new(config);
        
        // Create thread pool
        let worker_pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(config.parallel_config.worker_threads)
                .build()?
        );
        
        let buffers = Arc::new(RwLock::new(HashMap::new()));
        let processing_metrics = Arc::new(RwLock::new(StreamProcessingMetrics::default()));
        let state = Arc::new(RwLock::new(StreamProcessingState::default()));
        
        Ok(Self {
            base,
            config,
            buffers,
            processing_metrics,
            state,
            worker_pool,
        })
    }
    
    /// Process stream data
    pub async fn process_stream(&self, stream_id: &str, data: StreamDataItem) -> Result<Vec<StreamDataItem>> {
        let start_time = Instant::now();
        
        // Add to buffer
        {
            let mut buffers = self.buffers.write().await;
            let buffer = buffers.entry(stream_id.to_string())
                .or_insert_with(|| AdaptiveBuffer::new(self.config.buffer_config.adaptive_buffering.clone()));
            
            buffer.push(data)?;
        }
        
        // Process if batch is ready
        let batch = self.get_processing_batch(stream_id).await?;
        
        if batch.is_empty() {
            return Ok(Vec::new());
        }
        
        // Process batch in parallel
        let processed_items = self.process_batch_parallel(batch).await?;
        
        // Update metrics
        let processing_time = start_time.elapsed().as_micros() as f64;
        {
            let mut metrics = self.processing_metrics.write().await;
            metrics.messages_processed += processed_items.len() as u64;
            metrics.average_processing_time_us = 
                (metrics.average_processing_time_us + processing_time) / 2.0;
            
            if processing_time > metrics.max_processing_time_us {
                metrics.max_processing_time_us = processing_time;
            }
            
            metrics.last_update = chrono::Utc::now();
        }
        
        Ok(processed_items)
    }
    
    /// Get processing batch
    async fn get_processing_batch(&self, stream_id: &str) -> Result<Vec<StreamDataItem>> {
        let mut batch = Vec::new();
        
        let mut buffers = self.buffers.write().await;
        if let Some(buffer) = buffers.get_mut(stream_id) {
            let batch_size = self.config.processing_config.batch_size.min(buffer.len());
            
            for _ in 0..batch_size {
                if let Some(item) = buffer.pop() {
                    batch.push(item);
                } else {
                    break;
                }
            }
        }
        
        Ok(batch)
    }
    
    /// Process batch in parallel
    async fn process_batch_parallel(&self, batch: Vec<StreamDataItem>) -> Result<Vec<StreamDataItem>> {
        let config = self.config.clone();
        let worker_pool = self.worker_pool.clone();
        
        // Use rayon for parallel processing
        let processed: Vec<StreamDataItem> = worker_pool.install(|| {
            batch.into_par_iter()
                .map(|item| Self::process_single_item(item, &config))
                .collect::<Result<Vec<_>>>()
        })?;
        
        Ok(processed)
    }
    
    /// Process single item
    fn process_single_item(item: StreamDataItem, config: &StreamProcessingConfig) -> Result<StreamDataItem> {
        // Apply transformations based on configuration
        let mut processed_data = item.data.clone();
        
        // SIMD optimizations would be applied here for numeric data
        if config.simd_config.enabled {
            processed_data = Self::apply_simd_optimizations(processed_data, &config.simd_config)?;
        }
        
        // Apply aggregations
        processed_data = Self::apply_aggregations(processed_data, &config.processing_config.aggregation_config)?;
        
        Ok(StreamDataItem {
            id: uuid::Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            source: format!("processed_{}", item.source),
            data: processed_data,
            metadata: item.metadata,
        })
    }
    
    /// Apply SIMD optimizations
    fn apply_simd_optimizations(data: serde_json::Value, config: &SimdConfig) -> Result<serde_json::Value> {
        // SIMD operations would be implemented here
        // For now, return the data as-is
        Ok(data)
    }
    
    /// Apply aggregations
    fn apply_aggregations(data: serde_json::Value, config: &AggregationConfig) -> Result<serde_json::Value> {
        let mut result = serde_json::Map::new();
        
        if let Some(obj) = data.as_object() {
            for function in &config.functions {
                match function {
                    AggregationFunction::Count => {
                        result.insert("count".to_string(), serde_json::Value::Number(
                            serde_json::Number::from(obj.len())
                        ));
                    }
                    AggregationFunction::Sum => {
                        let sum: f64 = obj.values()
                            .filter_map(|v| v.as_f64())
                            .sum();
                        result.insert("sum".to_string(), serde_json::Value::Number(
                            serde_json::Number::from_f64(sum).unwrap_or(serde_json::Number::from(0))
                        ));
                    }
                    AggregationFunction::Average => {
                        let values: Vec<f64> = obj.values()
                            .filter_map(|v| v.as_f64())
                            .collect();
                        
                        if !values.is_empty() {
                            let avg = values.iter().sum::<f64>() / values.len() as f64;
                            result.insert("average".to_string(), serde_json::Value::Number(
                                serde_json::Number::from_f64(avg).unwrap_or(serde_json::Number::from(0))
                            ));
                        }
                    }
                    _ => {
                        // Other aggregation functions would be implemented here
                    }
                }
            }
        }
        
        Ok(serde_json::Value::Object(result))
    }
    
    /// Get stream processing metrics
    pub async fn get_stream_metrics(&self) -> StreamProcessingMetrics {
        self.processing_metrics.read().await.clone()
    }
    
    /// Get stream processing state
    pub async fn get_stream_state(&self) -> StreamProcessingState {
        self.state.read().await.clone()
    }
    
    /// Get buffer statistics
    pub async fn get_buffer_stats(&self) -> HashMap<String, BufferStats> {
        let buffers = self.buffers.read().await;
        let mut stats = HashMap::new();
        
        for (stream_id, buffer) in buffers.iter() {
            stats.insert(stream_id.clone(), BufferStats {
                size: buffer.len(),
                capacity: buffer.current_size,
                utilization: buffer.len() as f64 / buffer.current_size as f64,
            });
        }
        
        stats
    }
}

/// Buffer statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferStats {
    pub size: usize,
    pub capacity: usize,
    pub utilization: f64,
}

#[async_trait]
impl DataAgent for StreamProcessingAgent {
    fn get_id(&self) -> DataAgentId {
        self.base.id
    }
    
    fn get_type(&self) -> DataAgentType {
        DataAgentType::StreamProcessing
    }
    
    async fn get_state(&self) -> DataAgentState {
        self.base.state.read().await.clone()
    }
    
    async fn get_info(&self) -> DataAgentInfo {
        self.base.info.read().await.clone()
    }
    
    async fn start(&self) -> Result<()> {
        info!("Starting stream processing agent");
        
        self.base.update_state(DataAgentState::Running).await?;
        
        // Update state
        {
            let mut state = self.state.write().await;
            state.active_workers = self.config.parallel_config.worker_threads;
        }
        
        info!("Stream processing agent started successfully");
        Ok(())
    }
    
    async fn stop(&self) -> Result<()> {
        info!("Stopping stream processing agent");
        
        self.base.update_state(DataAgentState::Stopping).await?;
        
        // Clear buffers
        self.buffers.write().await.clear();
        
        // Update state
        {
            let mut state = self.state.write().await;
            state.active_streams = 0;
            state.active_workers = 0;
        }
        
        self.base.update_state(DataAgentState::Stopped).await?;
        
        info!("Stream processing agent stopped successfully");
        Ok(())
    }
    
    async fn process(&self, message: DataMessage) -> Result<DataMessage> {
        let start_time = Instant::now();
        
        // Convert message to stream data item
        let stream_item = StreamDataItem {
            id: message.id,
            timestamp: message.timestamp,
            source: message.source.to_string(),
            data: message.payload,
            metadata: StreamMetadata {
                stream_id: "default".to_string(),
                sequence_number: 0,
                partition_key: "default".to_string(),
                priority: message.metadata.priority,
                watermark: message.timestamp,
            },
        };
        
        // Process the stream
        let processed_items = self.process_stream("default", stream_item).await?;
        
        // Update metrics
        let latency = start_time.elapsed().as_micros() as f64;
        self.base.update_metrics(MetricsUpdate::MessageProcessed(latency)).await?;
        
        // Create response message
        let response_payload = if processed_items.is_empty() {
            serde_json::json!({"status": "buffered"})
        } else {
            serde_json::to_value(processed_items)?
        };
        
        let response = DataMessage {
            id: uuid::Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            source: self.get_id(),
            destination: message.destination,
            message_type: DataMessageType::StreamData,
            payload: response_payload,
            metadata: MessageMetadata {
                priority: MessagePriority::High,
                expires_at: Some(chrono::Utc::now() + chrono::Duration::seconds(30)),
                retry_count: 0,
                trace_id: format!("stream_processing_{}", uuid::Uuid::new_v4()),
                span_id: format!("span_{}", uuid::Uuid::new_v4()),
            },
        };
        
        Ok(response)
    }
    
    async fn health_check(&self) -> Result<HealthStatus> {
        let state = self.get_stream_state().await;
        let metrics = self.get_stream_metrics().await;
        
        let health_level = if state.is_healthy {
            HealthLevel::Healthy
        } else {
            HealthLevel::Critical
        };
        
        Ok(HealthStatus {
            status: health_level,
            last_check: chrono::Utc::now(),
            uptime: self.base.start_time.elapsed(),
            issues: Vec::new(),
            metrics: HealthMetrics {
                cpu_usage_percent: metrics.worker_utilization,
                memory_usage_mb: 0.0, // Would be measured
                network_usage_mbps: 0.0, // Would be measured
                disk_usage_mb: 0.0, // Would be measured
                error_rate: metrics.messages_dropped as f64 / metrics.messages_processed.max(1) as f64,
                response_time_ms: metrics.average_processing_time_us / 1000.0,
            },
        })
    }
    
    async fn get_metrics(&self) -> Result<AgentMetrics> {
        Ok(self.base.metrics.read().await.clone())
    }
    
    async fn reset(&self) -> Result<()> {
        info!("Resetting stream processing agent");
        
        self.buffers.write().await.clear();
        
        // Reset metrics
        {
            let mut metrics = self.processing_metrics.write().await;
            *metrics = StreamProcessingMetrics::default();
        }
        
        // Reset state
        {
            let mut state = self.state.write().await;
            *state = StreamProcessingState::default();
        }
        
        info!("Stream processing agent reset successfully");
        Ok(())
    }
    
    async fn handle_coordination(&self, message: CoordinationMessage) -> Result<()> {
        debug!("Handling coordination message: {:?}", message.coordination_type);
        
        match message.coordination_type {
            crate::agents::base::CoordinationType::LoadBalancing => {
                info!("Received load balancing coordination");
            }
            crate::agents::base::CoordinationType::StateSync => {
                info!("Received state sync coordination");
            }
            _ => {
                debug!("Unhandled coordination type: {:?}", message.coordination_type);
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;
    
    #[test]
    async fn test_stream_processing_agent_creation() {
        let config = StreamProcessingConfig::default();
        let agent = StreamProcessingAgent::new(config).await;
        assert!(agent.is_ok());
    }
    
    #[test]
    async fn test_adaptive_buffer() {
        let config = AdaptiveBufferingConfig::default();
        let mut buffer = AdaptiveBuffer::new(config);
        
        // Test basic operations
        assert!(buffer.push(1).is_ok());
        assert_eq!(buffer.len(), 1);
        
        let item = buffer.pop();
        assert_eq!(item, Some(1));
        assert!(buffer.is_empty());
    }
    
    #[test]
    async fn test_stream_processing() {
        let config = StreamProcessingConfig::default();
        let agent = StreamProcessingAgent::new(config).await.unwrap();
        
        let stream_item = StreamDataItem {
            id: uuid::Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            source: "test".to_string(),
            data: serde_json::json!({"value": 42.0}),
            metadata: StreamMetadata {
                stream_id: "test_stream".to_string(),
                sequence_number: 1,
                partition_key: "test".to_string(),
                priority: MessagePriority::Normal,
                watermark: chrono::Utc::now(),
            },
        };
        
        let result = agent.process_stream("test_stream", stream_item).await;
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_pid_controller() {
        let mut controller = PIDController::new(1.0, 0.1, 0.01);
        controller.setpoint = 50.0;
        
        let output = controller.update(45.0, 0.1);
        assert!(output > 0.0); // Should try to increase to reach setpoint
    }
}