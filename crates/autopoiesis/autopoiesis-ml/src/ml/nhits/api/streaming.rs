use anyhow::Result;
use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::{Duration, Instant},
};
use tokio::{
    sync::{broadcast, mpsc, RwLock},
    time::interval,
};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::ml::nhits::api::{
    models::*,
    websocket::{send_forecast_update, WsMessage},
};

/// Streaming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Buffer size for incoming data
    pub buffer_size: usize,
    /// Batch size for processing
    pub batch_size: usize,
    /// Processing interval in milliseconds
    pub processing_interval_ms: u64,
    /// Maximum latency before forced processing
    pub max_latency_ms: u64,
    /// Enable automatic scaling
    pub auto_scaling: bool,
    /// Maximum concurrent streams
    pub max_concurrent_streams: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 10000,
            batch_size: 100,
            processing_interval_ms: 100,
            max_latency_ms: 1000,
            auto_scaling: true,
            max_concurrent_streams: 100,
        }
    }
}

/// Real-time data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub values: Vec<f64>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

impl DataPoint {
    pub fn new(values: Vec<f64>) -> Self {
        Self {
            timestamp: chrono::Utc::now(),
            values,
            metadata: None,
        }
    }

    pub fn with_timestamp(mut self, timestamp: chrono::DateTime<chrono::Utc>) -> Self {
        self.timestamp = timestamp;
        self
    }

    pub fn with_metadata(mut self, metadata: HashMap<String, serde_json::Value>) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Streaming forecast result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingForecastResult {
    pub predictions: Vec<f64>,
    pub confidence_intervals: Option<(Vec<f64>, Vec<f64>)>, // (lower, upper)
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub latency_ms: f64,
    pub model_version: String,
}

/// Data stream for real-time processing
pub struct DataStream {
    id: Uuid,
    buffer: Arc<RwLock<VecDeque<DataPoint>>>,
    config: StreamingConfig,
    processors: Vec<Arc<dyn StreamProcessor + Send + Sync>>,
    subscribers: Arc<RwLock<HashMap<Uuid, mpsc::UnboundedSender<StreamingForecastResult>>>>,
    metrics: StreamingMetrics,
    is_active: Arc<RwLock<bool>>,
}

impl DataStream {
    /// Create a new data stream
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            buffer: Arc::new(RwLock::new(VecDeque::new())),
            config,
            processors: Vec::new(),
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            metrics: StreamingMetrics::new(),
            is_active: Arc::new(RwLock::new(false)),
        }
    }

    /// Add a data point to the stream
    pub async fn push(&self, data_point: DataPoint) -> Result<()> {
        let mut buffer = self.buffer.write().await;
        
        // Maintain buffer size limit
        if buffer.len() >= self.config.buffer_size {
            buffer.pop_front();
            self.metrics.increment_dropped_points().await;
        }
        
        buffer.push_back(data_point);
        self.metrics.increment_received_points().await;
        
        Ok(())
    }

    /// Add multiple data points to the stream
    pub async fn push_batch(&self, data_points: Vec<DataPoint>) -> Result<()> {
        let mut buffer = self.buffer.write().await;
        
        for point in data_points {
            if buffer.len() >= self.config.buffer_size {
                buffer.pop_front();
                self.metrics.increment_dropped_points().await;
            }
            
            buffer.push_back(point);
            self.metrics.increment_received_points().await;
        }
        
        Ok(())
    }

    /// Subscribe to stream results
    pub async fn subscribe(&self) -> (Uuid, mpsc::UnboundedReceiver<StreamingForecastResult>) {
        let subscription_id = Uuid::new_v4();
        let (tx, rx) = mpsc::unbounded_channel();
        
        let mut subscribers = self.subscribers.write().await;
        subscribers.insert(subscription_id, tx);
        
        (subscription_id, rx)
    }

    /// Unsubscribe from stream results
    pub async fn unsubscribe(&self, subscription_id: Uuid) {
        let mut subscribers = self.subscribers.write().await;
        subscribers.remove(&subscription_id);
    }

    /// Add a stream processor
    pub fn add_processor<P: StreamProcessor + Send + Sync + 'static>(&mut self, processor: P) {
        self.processors.push(Arc::new(processor));
    }

    /// Start processing the stream
    pub async fn start(&self) -> Result<()> {
        {
            let mut is_active = self.is_active.write().await;
            if *is_active {
                return Ok(());
            }
            *is_active = true;
        }

        info!("Starting data stream: {}", self.id);

        // Start processing loop
        let buffer = self.buffer.clone();
        let config = self.config.clone();
        let processors = self.processors.clone();
        let subscribers = self.subscribers.clone();
        let metrics = self.metrics.clone();
        let is_active = self.is_active.clone();

        tokio::spawn(async move {
            let mut processing_interval = interval(Duration::from_millis(config.processing_interval_ms));
            let mut last_process_time = Instant::now();

            loop {
                // Check if stream is still active
                {
                    let active = is_active.read().await;
                    if !*active {
                        break;
                    }
                }

                processing_interval.tick().await;

                let should_process = {
                    let buffer = buffer.read().await;
                    buffer.len() >= config.batch_size ||
                    (buffer.len() > 0 && last_process_time.elapsed().as_millis() >= config.max_latency_ms as u128)
                };

                if should_process {
                    let batch = {
                        let mut buffer = buffer.write().await;
                        let batch_size = config.batch_size.min(buffer.len());
                        buffer.drain(..batch_size).collect::<Vec<_>>()
                    };

                    if !batch.is_empty() {
                        let start_time = Instant::now();

                        // Process batch with all processors
                        for processor in &processors {
                            if let Err(e) = Self::process_batch(&**processor, &batch, &subscribers, &metrics).await {
                                error!("Processing error: {}", e);
                                metrics.increment_processing_errors().await;
                            }
                        }

                        let processing_time = start_time.elapsed();
                        metrics.record_processing_time(processing_time).await;
                        last_process_time = Instant::now();

                        debug!("Processed batch of {} points in {:?}", batch.len(), processing_time);
                    }
                }

                // Update throughput metrics
                metrics.update_throughput().await;
            }

            info!("Data stream processing stopped: {}", Uuid::new_v4());
        });

        Ok(())
    }

    /// Stop processing the stream
    pub async fn stop(&self) {
        let mut is_active = self.is_active.write().await;
        *is_active = false;
        info!("Stopping data stream: {}", self.id);
    }

    /// Process a batch of data points
    async fn process_batch(
        processor: &dyn StreamProcessor,
        batch: &[DataPoint],
        subscribers: &Arc<RwLock<HashMap<Uuid, mpsc::UnboundedSender<StreamingForecastResult>>>>,
        metrics: &StreamingMetrics,
    ) -> Result<()> {
        let results = processor.process_batch(batch).await?;

        // Send results to subscribers
        let subscribers = subscribers.read().await;
        for result in results {
            for (_, sender) in subscribers.iter() {
                if let Err(e) = sender.send(result.clone()) {
                    warn!("Failed to send result to subscriber: {}", e);
                }
            }
            metrics.increment_processed_points().await;
        }

        Ok(())
    }

    /// Get stream statistics
    pub async fn get_stats(&self) -> StreamStats {
        let buffer_size = self.buffer.read().await.len();
        let subscriber_count = self.subscribers.read().await.len();

        StreamStats {
            stream_id: self.id,
            buffer_size,
            subscriber_count,
            is_active: *self.is_active.read().await,
            metrics: self.metrics.get_summary().await,
        }
    }
}

/// Stream processor trait
#[async_trait::async_trait]
pub trait StreamProcessor {
    async fn process_batch(&self, batch: &[DataPoint]) -> Result<Vec<StreamingForecastResult>>;
}

/// NHITS streaming forecaster
pub struct NHITSStreamProcessor {
    model: Arc<dyn crate::ml::nhits::NHITSModelTrait + Send + Sync>,
    model_id: String,
    config: ModelConfig,
    window_size: usize,
}

impl NHITSStreamProcessor {
    pub fn new(
        model: Arc<dyn crate::ml::nhits::NHITSModelTrait + Send + Sync>,
        model_id: String,
        config: ModelConfig,
    ) -> Self {
        Self {
            model,
            model_id,
            window_size: config.input_size,
            config,
        }
    }
}

#[async_trait::async_trait]
impl StreamProcessor for NHITSStreamProcessor {
    async fn process_batch(&self, batch: &[DataPoint]) -> Result<Vec<StreamingForecastResult>> {
        let mut results = Vec::new();

        if batch.len() < self.window_size {
            return Ok(results);
        }

        let start_time = Instant::now();

        // Extract features from the batch
        let features: Vec<Vec<f64>> = batch
            .iter()
            .map(|point| point.values.clone())
            .collect();

        // Use sliding window approach for continuous forecasting
        for i in self.window_size..=batch.len() {
            let window_start = i - self.window_size;
            let input_window: Vec<Vec<f64>> = features[window_start..i].to_vec();

            // Generate forecast
            match self.model.forecast(&input_window, self.config.output_size).await {
                Ok(predictions) => {
                    let result = StreamingForecastResult {
                        predictions,
                        confidence_intervals: None, // Could add uncertainty estimation
                        timestamp: batch[i-1].timestamp,
                        latency_ms: start_time.elapsed().as_millis() as f64,
                        model_version: "1.0.0".to_string(),
                    };
                    results.push(result);
                }
                Err(e) => {
                    error!("Forecast error: {}", e);
                }
            }
        }

        Ok(results)
    }
}

/// Stream manager for handling multiple data streams
pub struct StreamManager {
    streams: Arc<RwLock<HashMap<Uuid, Arc<DataStream>>>>,
    config: StreamingConfig,
    global_metrics: Arc<StreamingMetrics>,
}

impl StreamManager {
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            streams: Arc::new(RwLock::new(HashMap::new())),
            config,
            global_metrics: Arc::new(StreamingMetrics::new()),
        }
    }

    /// Create a new data stream
    pub async fn create_stream(&self) -> Result<Uuid> {
        let stream = Arc::new(DataStream::new(self.config.clone()));
        let stream_id = stream.id;

        let mut streams = self.streams.write().await;
        if streams.len() >= self.config.max_concurrent_streams {
            return Err(anyhow::anyhow!("Maximum number of concurrent streams reached"));
        }

        streams.insert(stream_id, stream);
        Ok(stream_id)
    }

    /// Get a stream by ID
    pub async fn get_stream(&self, stream_id: Uuid) -> Option<Arc<DataStream>> {
        let streams = self.streams.read().await;
        streams.get(&stream_id).cloned()
    }

    /// Remove a stream
    pub async fn remove_stream(&self, stream_id: Uuid) -> Result<()> {
        if let Some(stream) = self.get_stream(stream_id).await {
            stream.stop().await;
        }

        let mut streams = self.streams.write().await;
        streams.remove(&stream_id);
        Ok(())
    }

    /// List all streams
    pub async fn list_streams(&self) -> Vec<StreamStats> {
        let streams = self.streams.read().await;
        let mut stats = Vec::new();

        for stream in streams.values() {
            stats.push(stream.get_stats().await);
        }

        stats
    }

    /// Get global streaming metrics
    pub async fn get_global_metrics(&self) -> StreamingMetrics {
        self.global_metrics.clone()
    }
}

/// Streaming metrics
#[derive(Debug, Clone)]
pub struct StreamingMetrics {
    received_points: Arc<RwLock<u64>>,
    processed_points: Arc<RwLock<u64>>,
    dropped_points: Arc<RwLock<u64>>,
    processing_errors: Arc<RwLock<u64>>,
    processing_times: Arc<RwLock<VecDeque<Duration>>>,
    throughput_history: Arc<RwLock<VecDeque<(Instant, u64)>>>,
}

impl StreamingMetrics {
    pub fn new() -> Self {
        Self {
            received_points: Arc::new(RwLock::new(0)),
            processed_points: Arc::new(RwLock::new(0)),
            dropped_points: Arc::new(RwLock::new(0)),
            processing_errors: Arc::new(RwLock::new(0)),
            processing_times: Arc::new(RwLock::new(VecDeque::new())),
            throughput_history: Arc::new(RwLock::new(VecDeque::new())),
        }
    }

    pub async fn increment_received_points(&self) {
        let mut count = self.received_points.write().await;
        *count += 1;
    }

    pub async fn increment_processed_points(&self) {
        let mut count = self.processed_points.write().await;
        *count += 1;
    }

    pub async fn increment_dropped_points(&self) {
        let mut count = self.dropped_points.write().await;
        *count += 1;
    }

    pub async fn increment_processing_errors(&self) {
        let mut count = self.processing_errors.write().await;
        *count += 1;
    }

    pub async fn record_processing_time(&self, duration: Duration) {
        let mut times = self.processing_times.write().await;
        times.push_back(duration);
        
        // Keep only last 100 measurements
        if times.len() > 100 {
            times.pop_front();
        }
    }

    pub async fn update_throughput(&self) {
        let now = Instant::now();
        let processed = *self.processed_points.read().await;
        
        let mut history = self.throughput_history.write().await;
        history.push_back((now, processed));
        
        // Keep only last minute of data
        let cutoff = now - Duration::from_secs(60);
        while let Some((timestamp, _)) = history.front() {
            if *timestamp < cutoff {
                history.pop_front();
            } else {
                break;
            }
        }
    }

    pub async fn get_summary(&self) -> StreamingMetricsSummary {
        let received = *self.received_points.read().await;
        let processed = *self.processed_points.read().await;
        let dropped = *self.dropped_points.read().await;
        let errors = *self.processing_errors.read().await;
        
        let avg_processing_time = {
            let times = self.processing_times.read().await;
            if times.is_empty() {
                Duration::ZERO
            } else {
                let total: Duration = times.iter().sum();
                total / times.len() as u32
            }
        };

        let throughput = {
            let history = self.throughput_history.read().await;
            if history.len() < 2 {
                0.0
            } else {
                let (start_time, start_count) = history.front().unwrap();
                let (end_time, end_count) = history.back().unwrap();
                let duration = end_time.duration_since(*start_time).as_secs_f64();
                if duration > 0.0 {
                    (*end_count - *start_count) as f64 / duration
                } else {
                    0.0
                }
            }
        };

        StreamingMetricsSummary {
            received_points: received,
            processed_points: processed,
            dropped_points: dropped,
            processing_errors: errors,
            average_processing_time_ms: avg_processing_time.as_millis() as f64,
            throughput_per_second: throughput,
            success_rate: if received > 0 { (processed as f64) / (received as f64) } else { 0.0 },
        }
    }
}

/// Stream statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct StreamStats {
    pub stream_id: Uuid,
    pub buffer_size: usize,
    pub subscriber_count: usize,
    pub is_active: bool,
    pub metrics: StreamingMetricsSummary,
}

/// Streaming metrics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingMetricsSummary {
    pub received_points: u64,
    pub processed_points: u64,
    pub dropped_points: u64,
    pub processing_errors: u64,
    pub average_processing_time_ms: f64,
    pub throughput_per_second: f64,
    pub success_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_data_stream_creation() {
        let config = StreamingConfig::default();
        let stream = DataStream::new(config);
        
        assert!(!*stream.is_active.read().await);
        assert_eq!(stream.buffer.read().await.len(), 0);
    }

    #[tokio::test]
    async fn test_data_point_creation() {
        let point = DataPoint::new(vec![1.0, 2.0, 3.0])
            .with_metadata({
                let mut metadata = HashMap::new();
                metadata.insert("source".to_string(), serde_json::Value::String("test".to_string()));
                metadata
            });
        
        assert_eq!(point.values, vec![1.0, 2.0, 3.0]);
        assert!(point.metadata.is_some());
    }

    #[tokio::test]
    async fn test_stream_push() {
        let config = StreamingConfig::default();
        let stream = DataStream::new(config);
        
        let point = DataPoint::new(vec![1.0, 2.0]);
        stream.push(point).await.unwrap();
        
        assert_eq!(stream.buffer.read().await.len(), 1);
    }

    #[tokio::test]
    async fn test_stream_subscription() {
        let config = StreamingConfig::default();
        let stream = DataStream::new(config);
        
        let (subscription_id, mut rx) = stream.subscribe().await;
        assert!(stream.subscribers.read().await.contains_key(&subscription_id));
        
        stream.unsubscribe(subscription_id).await;
        assert!(!stream.subscribers.read().await.contains_key(&subscription_id));
    }

    #[tokio::test]
    async fn test_streaming_metrics() {
        let metrics = StreamingMetrics::new();
        
        metrics.increment_received_points().await;
        metrics.increment_processed_points().await;
        
        let summary = metrics.get_summary().await;
        assert_eq!(summary.received_points, 1);
        assert_eq!(summary.processed_points, 1);
        assert_eq!(summary.success_rate, 1.0);
    }
}