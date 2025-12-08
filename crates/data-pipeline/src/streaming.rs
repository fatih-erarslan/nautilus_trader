//! Real-time streaming data framework with Apache Kafka integration

use crate::{config::StreamingConfig, error::DataPipelineError, types::DataItem, ComponentHealth};
use std::sync::Arc;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc, Mutex};
use tokio::time::{sleep, timeout};
#[cfg(feature = "kafka")]
use rdkafka::config::ClientConfig;
#[cfg(feature = "kafka")]
use rdkafka::consumer::{Consumer, StreamConsumer};
#[cfg(feature = "kafka")]
use rdkafka::producer::{FutureProducer, FutureRecord};
#[cfg(feature = "kafka")]
use rdkafka::message::{Message, BorrowedMessage};
#[cfg(feature = "kafka")]
use rdkafka::util::Timeout;
use serde::{Deserialize, Serialize};
use tracing::{info, debug, warn, error, instrument};
use anyhow::Result;
use futures::stream::StreamExt;
use arrow::record_batch::RecordBatch;
use arrow::array::{Float64Array, StringArray, TimestampNanosecondArray};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
// use polars::prelude::*;
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// High-performance streaming engine with Apache Kafka integration
pub struct StreamingEngine {
    config: Arc<StreamingConfig>,
    #[cfg(feature = "kafka")]
    consumer: Arc<Mutex<Option<StreamConsumer>>>,
    #[cfg(feature = "kafka")]
    producer: Arc<Mutex<Option<FutureProducer>>>,
    data_sender: Arc<Mutex<Option<mpsc::UnboundedSender<StreamData>>>>,
    state: Arc<RwLock<StreamingState>>,
    metrics: Arc<RwLock<StreamingMetrics>>,
    buffer_pool: Arc<BufferPool>,
    schema_registry: Arc<SchemaRegistry>,
    compression_engine: Arc<CompressionEngine>,
    partitioner: Arc<Partitioner>,
}

/// Streaming state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingState {
    pub is_running: bool,
    pub consumer_offset: HashMap<String, i64>,
    pub producer_offset: HashMap<String, i64>,
    pub last_message_time: Option<chrono::DateTime<chrono::Utc>>,
    pub connection_status: ConnectionStatus,
    pub error_count: u32,
    pub restart_count: u32,
}

impl Default for StreamingState {
    fn default() -> Self {
        Self {
            is_running: false,
            consumer_offset: HashMap::new(),
            producer_offset: HashMap::new(),
            last_message_time: None,
            connection_status: ConnectionStatus::Disconnected,
            error_count: 0,
            restart_count: 0,
        }
    }
}

/// Connection status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectionStatus {
    Connected,
    Disconnected,
    Reconnecting,
    Error(String),
}

/// Streaming metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingMetrics {
    pub messages_consumed: u64,
    pub messages_produced: u64,
    pub bytes_consumed: u64,
    pub bytes_produced: u64,
    pub consumer_lag: HashMap<String, i64>,
    pub throughput_msg_per_sec: f64,
    pub throughput_bytes_per_sec: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub error_rate: f64,
    pub last_reset: chrono::DateTime<chrono::Utc>,
}

impl Default for StreamingMetrics {
    fn default() -> Self {
        Self {
            messages_consumed: 0,
            messages_produced: 0,
            bytes_consumed: 0,
            bytes_produced: 0,
            consumer_lag: HashMap::new(),
            throughput_msg_per_sec: 0.0,
            throughput_bytes_per_sec: 0.0,
            latency_p50_ms: 0.0,
            latency_p95_ms: 0.0,
            latency_p99_ms: 0.0,
            error_rate: 0.0,
            last_reset: chrono::Utc::now(),
        }
    }
}

/// Stream data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamData {
    pub topic: String,
    pub partition: i32,
    pub offset: i64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub key: Option<String>,
    pub payload: StreamPayload,
    pub headers: HashMap<String, String>,
    pub metadata: StreamMetadata,
}

/// Stream payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamPayload {
    MarketData(MarketData),
    NewsData(NewsData),
    SocialData(SocialData),
    EconomicData(EconomicData),
    TechnicalData(TechnicalData),
    Raw(Vec<u8>),
}

/// Market data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub symbol: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub trades: u64,
    pub vwap: f64,
    pub bid: f64,
    pub ask: f64,
    pub bid_size: f64,
    pub ask_size: f64,
}

/// News data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsData {
    pub id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub title: String,
    pub content: String,
    pub source: String,
    pub author: Option<String>,
    pub category: String,
    pub tags: Vec<String>,
    pub sentiment_score: Option<f64>,
    pub relevance_score: Option<f64>,
    pub symbols: Vec<String>,
}

/// Social media data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialData {
    pub id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub platform: String,
    pub content: String,
    pub author: String,
    pub followers: u64,
    pub likes: u64,
    pub shares: u64,
    pub comments: u64,
    pub hashtags: Vec<String>,
    pub mentions: Vec<String>,
    pub sentiment_score: Option<f64>,
    pub influence_score: Option<f64>,
}

/// Economic data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EconomicData {
    pub indicator: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub country: String,
    pub value: f64,
    pub previous_value: Option<f64>,
    pub forecast: Option<f64>,
    pub unit: String,
    pub frequency: String,
    pub importance: ImportanceLevel,
}

/// Technical data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalData {
    pub symbol: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub indicators: HashMap<String, f64>,
    pub patterns: Vec<PatternDetection>,
    pub signals: Vec<TradingSignal>,
}

/// Pattern detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternDetection {
    pub pattern_type: String,
    pub confidence: f64,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub end_time: chrono::DateTime<chrono::Utc>,
    pub parameters: HashMap<String, f64>,
}

/// Trading signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    pub signal_type: String,
    pub strength: f64,
    pub direction: SignalDirection,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub source: String,
    pub metadata: HashMap<String, String>,
}

/// Signal direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalDirection {
    Buy,
    Sell,
    Hold,
}

/// Importance level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImportanceLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Stream metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMetadata {
    pub source: String,
    pub version: String,
    pub schema_version: String,
    pub compression: Option<String>,
    pub checksum: Option<String>,
    pub processing_time: Option<Duration>,
    pub quality_score: Option<f64>,
}

/// Buffer pool for memory management
pub struct BufferPool {
    small_buffers: Arc<Mutex<Vec<Vec<u8>>>>,
    medium_buffers: Arc<Mutex<Vec<Vec<u8>>>>,
    large_buffers: Arc<Mutex<Vec<Vec<u8>>>>,
    small_buffer_size: usize,
    medium_buffer_size: usize,
    large_buffer_size: usize,
    max_buffers: usize,
}

impl BufferPool {
    pub fn new(
        small_size: usize,
        medium_size: usize,
        large_size: usize,
        max_buffers: usize,
    ) -> Self {
        Self {
            small_buffers: Arc::new(Mutex::new(Vec::new())),
            medium_buffers: Arc::new(Mutex::new(Vec::new())),
            large_buffers: Arc::new(Mutex::new(Vec::new())),
            small_buffer_size: small_size,
            medium_buffer_size: medium_size,
            large_buffer_size: large_size,
            max_buffers,
        }
    }

    pub async fn get_buffer(&self, size: usize) -> Vec<u8> {
        if size <= self.small_buffer_size {
            let mut buffers = self.small_buffers.lock().await;
            buffers.pop().unwrap_or_else(|| vec![0u8; self.small_buffer_size])
        } else if size <= self.medium_buffer_size {
            let mut buffers = self.medium_buffers.lock().await;
            buffers.pop().unwrap_or_else(|| vec![0u8; self.medium_buffer_size])
        } else {
            let mut buffers = self.large_buffers.lock().await;
            buffers.pop().unwrap_or_else(|| vec![0u8; self.large_buffer_size])
        }
    }

    pub async fn return_buffer(&self, mut buffer: Vec<u8>) {
        buffer.clear();
        
        if buffer.capacity() <= self.small_buffer_size {
            let mut buffers = self.small_buffers.lock().await;
            if buffers.len() < self.max_buffers {
                buffers.push(buffer);
            }
        } else if buffer.capacity() <= self.medium_buffer_size {
            let mut buffers = self.medium_buffers.lock().await;
            if buffers.len() < self.max_buffers {
                buffers.push(buffer);
            }
        } else {
            let mut buffers = self.large_buffers.lock().await;
            if buffers.len() < self.max_buffers {
                buffers.push(buffer);
            }
        }
    }
}

/// Schema registry for data validation
pub struct SchemaRegistry {
    schemas: Arc<RwLock<HashMap<String, Schema>>>,
}

impl SchemaRegistry {
    pub fn new() -> Self {
        Self {
            schemas: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn register_schema(&self, name: String, schema: Schema) {
        let mut schemas = self.schemas.write().await;
        schemas.insert(name, schema);
    }

    pub async fn get_schema(&self, name: &str) -> Option<Schema> {
        let schemas = self.schemas.read().await;
        schemas.get(name).cloned()
    }

    pub async fn validate_data(&self, schema_name: &str, data: &RecordBatch) -> Result<bool> {
        if let Some(schema) = self.get_schema(schema_name).await {
            Ok(data.schema().eq(&schema))
        } else {
            Ok(false)
        }
    }
}

/// Compression engine for data optimization
pub struct CompressionEngine {
    compression_type: CompressionType,
    compression_level: u32,
}

impl CompressionEngine {
    pub fn new(compression_type: CompressionType, level: u32) -> Self {
        Self {
            compression_type,
            compression_level: level,
        }
    }

    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        match self.compression_type {
            CompressionType::None => Ok(data.to_vec()),
            CompressionType::Gzip => {
                use flate2::write::GzEncoder;
                use flate2::Compression;
                use std::io::Write;
                
                let mut encoder = GzEncoder::new(Vec::new(), Compression::new(self.compression_level));
                encoder.write_all(data)?;
                Ok(encoder.finish()?)
            }
            CompressionType::Lz4 => {
                let compressed = lz4::block::compress(data, Some(lz4::block::CompressionMode::HIGHCOMPRESSION(self.compression_level as i32)), false)?;
                Ok(compressed)
            }
            CompressionType::Zstd => {
                let compressed = zstd::bulk::compress(data, self.compression_level as i32)?;
                Ok(compressed)
            }
            _ => Err(anyhow::anyhow!("Compression type not supported")),
        }
    }

    pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        match self.compression_type {
            CompressionType::None => Ok(data.to_vec()),
            CompressionType::Gzip => {
                use flate2::read::GzDecoder;
                use std::io::Read;
                
                let mut decoder = GzDecoder::new(data);
                let mut result = Vec::new();
                decoder.read_to_end(&mut result)?;
                Ok(result)
            }
            CompressionType::Lz4 => {
                let decompressed = lz4::block::decompress(data, None)?;
                Ok(decompressed)
            }
            CompressionType::Zstd => {
                let decompressed = zstd::bulk::decompress(data, 10 * 1024 * 1024)?; // 10MB max
                Ok(decompressed)
            }
            _ => Err(anyhow::anyhow!("Compression type not supported")),
        }
    }
}

/// Partitioner for load balancing
pub struct Partitioner {
    partition_count: usize,
    strategy: PartitionStrategy,
}

impl Partitioner {
    pub fn new(partition_count: usize, strategy: PartitionStrategy) -> Self {
        Self {
            partition_count,
            strategy,
        }
    }

    pub fn get_partition(&self, key: &str, data: &StreamData) -> usize {
        match self.strategy {
            PartitionStrategy::RoundRobin => {
                // Simple round-robin based on hash
                let hash = self.hash_key(key);
                (hash % self.partition_count as u64) as usize
            }
            PartitionStrategy::Hash => {
                let hash = self.hash_key(key);
                (hash % self.partition_count as u64) as usize
            }
            PartitionStrategy::Symbol => {
                // Partition by symbol for market data
                match &data.payload {
                    StreamPayload::MarketData(market_data) => {
                        let hash = self.hash_key(&market_data.symbol);
                        (hash % self.partition_count as u64) as usize
                    }
                    _ => self.get_partition(key, data),
                }
            }
            PartitionStrategy::Timestamp => {
                // Partition by timestamp for time-series data
                let timestamp = data.timestamp.timestamp() as u64;
                (timestamp % self.partition_count as u64) as usize
            }
        }
    }

    fn hash_key(&self, key: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }
}

/// Partition strategy
#[derive(Debug, Clone, Copy)]
pub enum PartitionStrategy {
    RoundRobin,
    Hash,
    Symbol,
    Timestamp,
}

use crate::config::CompressionType;

impl StreamingEngine {
    /// Create a new streaming engine
    pub async fn new(config: Arc<StreamingConfig>) -> Result<Self> {
        info!("Initializing streaming engine with config: {:?}", config);
        
        let buffer_pool = Arc::new(BufferPool::new(
            4 * 1024,     // 4KB small buffers
            64 * 1024,    // 64KB medium buffers
            1024 * 1024,  // 1MB large buffers
            1000,         // max 1000 buffers of each size
        ));
        
        let schema_registry = Arc::new(SchemaRegistry::new());
        let compression_engine = Arc::new(CompressionEngine::new(
            config.compression.clone(),
            6, // default compression level
        ));
        
        let partitioner = Arc::new(Partitioner::new(
            16, // 16 partitions
            PartitionStrategy::Hash,
        ));
        
        Ok(Self {
            config: config.clone(),
            #[cfg(feature = "kafka")]
            consumer: Arc::new(Mutex::new(None)),
            #[cfg(feature = "kafka")]
            producer: Arc::new(Mutex::new(None)),
            data_sender: Arc::new(Mutex::new(None)),
            state: Arc::new(RwLock::new(StreamingState::default())),
            metrics: Arc::new(RwLock::new(StreamingMetrics::default())),
            buffer_pool,
            schema_registry,
            compression_engine,
            partitioner,
        })
    }

    /// Start the streaming engine
    #[instrument(skip(self))]
    pub async fn start(&self) -> Result<()> {
        info!("Starting streaming engine");
        
        // Initialize consumer
        #[cfg(feature = "kafka")]
        self.initialize_consumer().await?;
        
        // Initialize producer
        #[cfg(feature = "kafka")]
        self.initialize_producer().await?;
        
        // Start consumer loop
        self.start_consumer_loop().await?;
        
        // Update state
        {
            let mut state = self.state.write().await;
            state.is_running = true;
            state.connection_status = ConnectionStatus::Connected;
        }
        
        info!("Streaming engine started successfully");
        Ok(())
    }

    /// Stop the streaming engine
    #[instrument(skip(self))]
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping streaming engine");
        
        // Update state
        {
            let mut state = self.state.write().await;
            state.is_running = false;
            state.connection_status = ConnectionStatus::Disconnected;
        }
        
        // Stop consumer
        #[cfg(feature = "kafka")]
        {
            let mut consumer = self.consumer.lock().await;
            *consumer = None;
        }
        
        // Stop producer
        #[cfg(feature = "kafka")]
        {
            let mut producer = self.producer.lock().await;
            *producer = None;
        }
        
        info!("Streaming engine stopped successfully");
        Ok(())
    }

    /// Initialize Kafka consumer
    #[cfg(feature = "kafka")]
    async fn initialize_consumer(&self) -> Result<()> {
        let mut client_config = ClientConfig::new();
        
        // Basic configuration
        client_config
            .set("bootstrap.servers", &self.config.kafka_brokers.join(","))
            .set("group.id", &self.config.consumer_group)
            .set("enable.partition.eof", "false")
            .set("session.timeout.ms", &self.config.session_timeout.as_millis().to_string())
            .set("heartbeat.interval.ms", &self.config.heartbeat_interval.as_millis().to_string())
            .set("max.poll.records", &self.config.max_poll_records.to_string())
            .set("fetch.min.bytes", &self.config.fetch_min_bytes.to_string())
            .set("fetch.max.wait.ms", &self.config.fetch_max_wait.as_millis().to_string())
            .set("auto.offset.reset", &format!("{:?}", self.config.auto_offset_reset).to_lowercase())
            .set("enable.auto.commit", &self.config.enable_auto_commit.to_string())
            .set("auto.commit.interval.ms", &self.config.auto_commit_interval.as_millis().to_string());
        
        // SSL configuration
        if self.config.enable_ssl {
            if let Some(ssl_config) = &self.config.ssl_config {
                client_config
                    .set("security.protocol", "SSL")
                    .set("ssl.ca.location", &ssl_config.ca_cert_path)
                    .set("ssl.certificate.location", &ssl_config.client_cert_path)
                    .set("ssl.key.location", &ssl_config.client_key_path)
                    .set("ssl.endpoint.identification.algorithm", if ssl_config.verify_hostname { "https" } else { "none" });
            }
        }
        
        // SASL configuration
        if let Some(sasl_config) = &self.config.sasl_config {
            client_config
                .set("security.protocol", "SASL_SSL")
                .set("sasl.mechanism", &format!("{:?}", sasl_config.mechanism).to_uppercase())
                .set("sasl.username", &sasl_config.username)
                .set("sasl.password", &sasl_config.password);
        }
        
        // Compression configuration
        client_config.set("compression.type", &format!("{:?}", self.config.compression).to_lowercase());
        
        // Create consumer
        let consumer: StreamConsumer = client_config.create()?;
        
        // Subscribe to topics
        consumer.subscribe(&self.config.topics.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
        
        // Store consumer
        {
            let mut consumer_guard = self.consumer.lock().await;
            *consumer_guard = Some(consumer);
        }
        
        info!("Kafka consumer initialized successfully");
        Ok(())
    }

    /// Initialize Kafka producer
    #[cfg(feature = "kafka")]
    async fn initialize_producer(&self) -> Result<()> {
        let mut client_config = ClientConfig::new();
        
        // Basic configuration
        client_config
            .set("bootstrap.servers", &self.config.kafka_brokers.join(","))
            .set("compression.type", &format!("{:?}", self.config.compression).to_lowercase())
            .set("batch.size", "16384")
            .set("linger.ms", "5")
            .set("buffer.memory", "33554432");
        
        // SSL configuration
        if self.config.enable_ssl {
            if let Some(ssl_config) = &self.config.ssl_config {
                client_config
                    .set("security.protocol", "SSL")
                    .set("ssl.ca.location", &ssl_config.ca_cert_path)
                    .set("ssl.certificate.location", &ssl_config.client_cert_path)
                    .set("ssl.key.location", &ssl_config.client_key_path)
                    .set("ssl.endpoint.identification.algorithm", if ssl_config.verify_hostname { "https" } else { "none" });
            }
        }
        
        // SASL configuration
        if let Some(sasl_config) = &self.config.sasl_config {
            client_config
                .set("security.protocol", "SASL_SSL")
                .set("sasl.mechanism", &format!("{:?}", sasl_config.mechanism).to_uppercase())
                .set("sasl.username", &sasl_config.username)
                .set("sasl.password", &sasl_config.password);
        }
        
        // Create producer
        let producer: FutureProducer = client_config.create()?;
        
        // Store producer
        {
            let mut producer_guard = self.producer.lock().await;
            *producer_guard = Some(producer);
        }
        
        info!("Kafka producer initialized successfully");
        Ok(())
    }

    /// Start consumer loop
    #[cfg(feature = "kafka")]
    async fn start_consumer_loop(&self) -> Result<()> {
        let (sender, mut receiver) = mpsc::unbounded_channel();
        
        // Store sender
        {
            let mut data_sender = self.data_sender.lock().await;
            *data_sender = Some(sender);
        }
        
        // Clone necessary components for the consumer task
        let consumer_arc = Arc::clone(&self.consumer);
        let state_arc = Arc::clone(&self.state);
        let metrics_arc = Arc::clone(&self.metrics);
        let buffer_pool_arc = Arc::clone(&self.buffer_pool);
        let compression_engine_arc = Arc::clone(&self.compression_engine);
        let batch_size = self.config.batch_size;
        let consumer_timeout = self.config.consumer_timeout;
        
        // Spawn consumer task
        tokio::spawn(async move {
            loop {
                // Check if we should continue running
                {
                    let state = state_arc.read().await;
                    if !state.is_running {
                        break;
                    }
                }
                
                // Get consumer
                let consumer = {
                    let consumer_guard = consumer_arc.lock().await;
                    consumer_guard.clone()
                };
                
                if let Some(consumer) = consumer {
                    // Poll for messages
                    match timeout(consumer_timeout, consumer.recv()).await {
                        Ok(Ok(message)) => {
                            // Process message
                            if let Err(e) = Self::process_message(
                                &message,
                                &metrics_arc,
                                &buffer_pool_arc,
                                &compression_engine_arc,
                            ).await {
                                error!("Error processing message: {}", e);
                                
                                // Update error count
                                {
                                    let mut state = state_arc.write().await;
                                    state.error_count += 1;
                                }
                            }
                        }
                        Ok(Err(e)) => {
                            error!("Consumer error: {}", e);
                            
                            // Update error count
                            {
                                let mut state = state_arc.write().await;
                                state.error_count += 1;
                                state.connection_status = ConnectionStatus::Error(e.to_string());
                            }
                        }
                        Err(_) => {
                            // Timeout - continue loop
                            debug!("Consumer timeout - continuing");
                        }
                    }
                } else {
                    // No consumer available - wait and retry
                    sleep(Duration::from_secs(1)).await;
                }
            }
        });
        
        info!("Consumer loop started");
        Ok(())
    }

    /// Start consumer loop (non-Kafka implementation)
    #[cfg(not(feature = "kafka"))]
    async fn start_consumer_loop(&self) -> Result<()> {
        info!("Starting non-Kafka consumer loop");
        
        // Create dummy sender for consistency
        let (sender, _receiver) = mpsc::unbounded_channel();
        
        // Store sender
        {
            let mut data_sender = self.data_sender.lock().await;
            *data_sender = Some(sender);
        }
        
        info!("Non-Kafka consumer loop started");
        Ok(())
    }

    /// Process a single message
    #[cfg(feature = "kafka")]
    async fn process_message(
        message: &BorrowedMessage<'_>,
        metrics: &Arc<RwLock<StreamingMetrics>>,
        buffer_pool: &Arc<BufferPool>,
        compression_engine: &Arc<CompressionEngine>,
    ) -> Result<()> {
        let start_time = Instant::now();
        
        // Extract message data
        let topic = message.topic().to_string();
        let partition = message.partition();
        let offset = message.offset();
        let timestamp = message.timestamp().to_millis().unwrap_or(0);
        let key = message.key().map(|k| String::from_utf8_lossy(k).to_string());
        
        // Get payload
        let payload = message.payload().ok_or_else(|| {
            anyhow::anyhow!("Message has no payload")
        })?;
        
        // Decompress if needed
        let decompressed_payload = compression_engine.decompress(payload)?;
        
        // Parse payload based on topic
        let stream_payload = Self::parse_payload(&topic, &decompressed_payload)?;
        
        // Create stream data
        let stream_data = StreamData {
            topic,
            partition,
            offset,
            timestamp: chrono::DateTime::from_timestamp(timestamp / 1000, 0)
                .unwrap_or_else(chrono::Utc::now),
            key,
            payload: stream_payload,
            headers: HashMap::new(), // TODO: Extract headers
            metadata: StreamMetadata {
                source: "kafka".to_string(),
                version: "1.0".to_string(),
                schema_version: "1.0".to_string(),
                compression: Some("auto".to_string()),
                checksum: None,
                processing_time: Some(start_time.elapsed()),
                quality_score: None,
            },
        };
        
        // Update metrics
        {
            let mut metrics = metrics.write().await;
            metrics.messages_consumed += 1;
            metrics.bytes_consumed += payload.len() as u64;
            metrics.latency_p99_ms = start_time.elapsed().as_millis() as f64;
        }
        
        debug!("Processed message: topic={}, partition={}, offset={}", 
               stream_data.topic, stream_data.partition, stream_data.offset);
        
        Ok(())
    }

    /// Parse payload based on topic
    fn parse_payload(topic: &str, payload: &[u8]) -> Result<StreamPayload> {
        let payload_str = String::from_utf8_lossy(payload);
        
        match topic {
            topic if topic.contains("market-data") => {
                let market_data: MarketData = serde_json::from_str(&payload_str)?;
                Ok(StreamPayload::MarketData(market_data))
            }
            topic if topic.contains("news") => {
                let news_data: NewsData = serde_json::from_str(&payload_str)?;
                Ok(StreamPayload::NewsData(news_data))
            }
            topic if topic.contains("social") => {
                let social_data: SocialData = serde_json::from_str(&payload_str)?;
                Ok(StreamPayload::SocialData(social_data))
            }
            topic if topic.contains("economic") => {
                let economic_data: EconomicData = serde_json::from_str(&payload_str)?;
                Ok(StreamPayload::EconomicData(economic_data))
            }
            topic if topic.contains("technical") => {
                let technical_data: TechnicalData = serde_json::from_str(&payload_str)?;
                Ok(StreamPayload::TechnicalData(technical_data))
            }
            _ => {
                Ok(StreamPayload::Raw(payload.to_vec()))
            }
        }
    }

    /// Produce a message to Kafka
    #[cfg(feature = "kafka")]
    pub async fn produce_message(&self, topic: &str, key: Option<&str>, payload: &[u8]) -> Result<()> {
        let producer = {
            let producer_guard = self.producer.lock().await;
            producer_guard.clone()
        };
        
        if let Some(producer) = producer {
            // Compress payload
            let compressed_payload = self.compression_engine.compress(payload)?;
            
            // Create record
            let mut record = FutureRecord::to(topic).payload(&compressed_payload);
            
            if let Some(key) = key {
                record = record.key(key);
            }
            
            // Send message
            match producer.send(record, Timeout::After(Duration::from_secs(5))).await {
                Ok((partition, offset)) => {
                    debug!("Message sent successfully: topic={}, partition={}, offset={}", 
                           topic, partition, offset);
                    
                    // Update metrics
                    {
                        let mut metrics = self.metrics.write().await;
                        metrics.messages_produced += 1;
                        metrics.bytes_produced += compressed_payload.len() as u64;
                    }
                    
                    Ok(())
                }
                Err((e, _)) => {
                    error!("Failed to send message: {}", e);
                    Err(anyhow::anyhow!("Failed to send message: {}", e))
                }
            }
        } else {
            Err(anyhow::anyhow!("Producer not available"))
        }
    }

    /// Produce a message (non-Kafka implementation)
    #[cfg(not(feature = "kafka"))]
    pub async fn produce_message(&self, topic: &str, key: Option<&str>, payload: &[u8]) -> Result<()> {
        debug!("Non-Kafka produce message to topic: {}, key: {:?}, payload size: {} bytes", 
               topic, key, payload.len());
        
        // Update metrics for consistency
        {
            let mut metrics = self.metrics.write().await;
            metrics.messages_produced += 1;
            metrics.bytes_produced += payload.len() as u64;
        }
        
        Ok(())
    }

    /// Get current state
    pub async fn get_state(&self) -> StreamingState {
        self.state.read().await.clone()
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> StreamingMetrics {
        self.metrics.read().await.clone()
    }

    /// Health check
    pub async fn health_check(&self) -> Result<ComponentHealth> {
        let state = self.state.read().await;
        
        match state.connection_status {
            ConnectionStatus::Connected => Ok(ComponentHealth::Healthy),
            ConnectionStatus::Disconnected => Ok(ComponentHealth::Unhealthy),
            ConnectionStatus::Reconnecting => Ok(ComponentHealth::Degraded),
            ConnectionStatus::Error(_) => Ok(ComponentHealth::Unhealthy),
        }
    }

    /// Reset streaming engine
    pub async fn reset(&self) -> Result<()> {
        info!("Resetting streaming engine");
        
        // Reset state
        {
            let mut state = self.state.write().await;
            *state = StreamingState::default();
        }
        
        // Reset metrics
        {
            let mut metrics = self.metrics.write().await;
            *metrics = StreamingMetrics::default();
        }
        
        info!("Streaming engine reset successfully");
        Ok(())
    }
}

// Convert DataItem to StreamData
impl From<DataItem> for StreamData {
    fn from(item: DataItem) -> Self {
        StreamData {
            topic: "unknown".to_string(),
            partition: 0,
            offset: 0,
            timestamp: chrono::Utc::now(),
            key: None,
            payload: StreamPayload::Raw(item.raw_data),
            headers: HashMap::new(),
            metadata: StreamMetadata {
                source: "internal".to_string(),
                version: "1.0".to_string(),
                schema_version: "1.0".to_string(),
                compression: None,
                checksum: None,
                processing_time: None,
                quality_score: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_streaming_engine_creation() {
        let config = Arc::new(StreamingConfig::default());
        let engine = StreamingEngine::new(config).await;
        assert!(engine.is_ok());
    }

    #[test]
    async fn test_buffer_pool() {
        let pool = BufferPool::new(1024, 4096, 16384, 10);
        
        let buffer = pool.get_buffer(512).await;
        assert!(buffer.len() >= 512);
        
        pool.return_buffer(buffer).await;
    }

    #[test]
    async fn test_compression() {
        let engine = CompressionEngine::new(CompressionType::Gzip, 6);
        
        let data = b"Hello, World!";
        let compressed = engine.compress(data).unwrap();
        let decompressed = engine.decompress(&compressed).unwrap();
        
        assert_eq!(data, decompressed.as_slice());
    }

    #[test]
    async fn test_partitioner() {
        let partitioner = Partitioner::new(4, PartitionStrategy::Hash);
        
        let stream_data = StreamData {
            topic: "test".to_string(),
            partition: 0,
            offset: 0,
            timestamp: chrono::Utc::now(),
            key: None,
            payload: StreamPayload::Raw(vec![]),
            headers: HashMap::new(),
            metadata: StreamMetadata {
                source: "test".to_string(),
                version: "1.0".to_string(),
                schema_version: "1.0".to_string(),
                compression: None,
                checksum: None,
                processing_time: None,
                quality_score: None,
            },
        };
        
        let partition = partitioner.get_partition("test-key", &stream_data);
        assert!(partition < 4);
    }
}