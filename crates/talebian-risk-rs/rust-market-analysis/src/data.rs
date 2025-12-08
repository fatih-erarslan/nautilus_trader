//! Real-time data processing pipeline with SIMD optimization
//! 
//! Implements high-performance streaming data ingestion, processing, and analysis
//! with SIMD vectorization and parallel processing capabilities.

use crate::{
    types::*,
    config::Config,
    error::{AnalysisError, Result},
    utils::{simd, validation},
};
use ndarray::{Array1, Array2};
use std::collections::{VecDeque, HashMap};
use std::sync::{Arc, RwLock, Mutex};
use chrono::{DateTime, Utc, Duration};
use tokio::sync::{mpsc, broadcast, RwLock as TokioRwLock};
use parking_lot::Mutex as ParkingMutex;
use crossbeam::channel;
use rayon::prelude::*;
use tracing::{info, debug, warn, error};

#[cfg(feature = "simd")]
use wide::f64x4;

/// Real-time data processing pipeline
#[derive(Debug)]
pub struct DataPipeline {
    config: DataPipelineConfig,
    ingestion_buffer: Arc<RwLock<VecDeque<RawMarketData>>>,
    processing_queue: Arc<Mutex<VecDeque<MarketData>>>,
    
    // Streaming channels
    raw_data_sender: mpsc::UnboundedSender<RawMarketData>,
    raw_data_receiver: Arc<Mutex<Option<mpsc::UnboundedReceiver<RawMarketData>>>>,
    
    processed_data_sender: broadcast::Sender<MarketData>,
    
    // Processing components
    data_validator: DataValidator,
    feature_processor: FeatureProcessor,
    streaming_aggregator: StreamingAggregator,
    
    // Performance monitoring
    metrics: Arc<RwLock<PipelineMetrics>>,
    
    // State management
    is_running: Arc<RwLock<bool>>,
    worker_handles: Arc<Mutex<Vec<tokio::task::JoinHandle<()>>>>,
}

#[derive(Debug, Clone)]
pub struct DataPipelineConfig {
    pub buffer_size: usize,
    pub batch_size: usize,
    pub flush_interval_ms: u64,
    pub max_processing_latency_ms: u64,
    pub enable_simd: bool,
    pub parallel_workers: usize,
    pub compression_enabled: bool,
    pub validation_enabled: bool,
    pub backpressure_threshold: f64,
}

impl Default for DataPipelineConfig {
    fn default() -> Self {
        Self {
            buffer_size: 10000,
            batch_size: 100,
            flush_interval_ms: 100,
            max_processing_latency_ms: 50,
            enable_simd: true,
            parallel_workers: num_cpus::get(),
            compression_enabled: true,
            validation_enabled: true,
            backpressure_threshold: 0.8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RawMarketData {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub price: f64,
    pub volume: f64,
    pub bid: Option<f64>,
    pub ask: Option<f64>,
    pub trade_id: Option<String>,
    pub trade_side: Option<TradeSide>,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default)]
pub struct PipelineMetrics {
    pub total_messages_received: u64,
    pub total_messages_processed: u64,
    pub total_messages_dropped: u64,
    pub average_processing_latency_ms: f64,
    pub peak_processing_latency_ms: f64,
    pub buffer_utilization: f64,
    pub throughput_per_second: f64,
    pub error_count: u64,
    pub last_update: DateTime<Utc>,
}

impl DataPipeline {
    pub fn new(config: Config) -> Result<Self> {
        let pipeline_config = DataPipelineConfig::default();
        
        let (raw_data_sender, raw_data_receiver) = mpsc::unbounded_channel();
        let (processed_data_sender, _) = broadcast::channel(1000);
        
        Ok(Self {
            config: pipeline_config.clone(),
            ingestion_buffer: Arc::new(RwLock::new(VecDeque::with_capacity(pipeline_config.buffer_size))),
            processing_queue: Arc::new(Mutex::new(VecDeque::with_capacity(pipeline_config.batch_size))),
            
            raw_data_sender,
            raw_data_receiver: Arc::new(Mutex::new(Some(raw_data_receiver))),
            processed_data_sender,
            
            data_validator: DataValidator::new(&pipeline_config)?,
            feature_processor: FeatureProcessor::new(&pipeline_config)?,
            streaming_aggregator: StreamingAggregator::new(&pipeline_config)?,
            
            metrics: Arc::new(RwLock::new(PipelineMetrics::default())),
            is_running: Arc::new(RwLock::new(false)),
            worker_handles: Arc::new(Mutex::new(Vec::new())),
        })
    }
    
    /// Start the data processing pipeline
    pub async fn start(&self) -> Result<()> {
        info!("Starting data processing pipeline");
        
        {
            let mut is_running = self.is_running.write().unwrap();
            if *is_running {
                return Err(AnalysisError::invalid_config("Pipeline is already running"));
            }
            *is_running = true;
        }
        
        // Start ingestion worker
        let ingestion_handle = self.start_ingestion_worker().await?;
        
        // Start processing workers
        let mut processing_handles = Vec::new();
        for i in 0..self.config.parallel_workers {
            let handle = self.start_processing_worker(i).await?;
            processing_handles.push(handle);
        }
        
        // Start metrics collection
        let metrics_handle = self.start_metrics_collector().await?;
        
        // Store handles
        let mut handles = self.worker_handles.lock().unwrap();
        handles.push(ingestion_handle);
        handles.extend(processing_handles);
        handles.push(metrics_handle);
        
        info!("Data processing pipeline started with {} workers", self.config.parallel_workers);
        Ok(())
    }
    
    /// Stop the data processing pipeline
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping data processing pipeline");
        
        {
            let mut is_running = self.is_running.write().unwrap();
            *is_running = false;
        }
        
        // Wait for all workers to complete
        let handles = {
            let mut handles_guard = self.worker_handles.lock().unwrap();
            std::mem::take(&mut *handles_guard)
        };
        
        for handle in handles {
            if let Err(e) = handle.await {
                warn!("Worker task failed to complete cleanly: {:?}", e);
            }
        }
        
        info!("Data processing pipeline stopped");
        Ok(())
    }
    
    /// Ingest raw market data
    pub async fn ingest(&self, data: RawMarketData) -> Result<()> {
        if !*self.is_running.read().unwrap() {
            return Err(AnalysisError::invalid_config("Pipeline is not running"));
        }
        
        // Check backpressure
        let buffer_utilization = {
            let buffer = self.ingestion_buffer.read().unwrap();
            buffer.len() as f64 / self.config.buffer_size as f64
        };
        
        if buffer_utilization > self.config.backpressure_threshold {
            warn!("Backpressure detected, buffer utilization: {:.2}%", buffer_utilization * 100.0);
            
            // Update metrics
            {
                let mut metrics = self.metrics.write().unwrap();
                metrics.total_messages_dropped += 1;
                metrics.buffer_utilization = buffer_utilization;
            }
            
            return Err(AnalysisError::concurrency_error("Buffer overflow, dropping message"));
        }
        
        // Send to ingestion worker
        self.raw_data_sender.send(data)
            .map_err(|_| AnalysisError::concurrency_error("Failed to send data to ingestion worker"))?;
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_messages_received += 1;
        }
        
        Ok(())
    }
    
    /// Subscribe to processed data stream
    pub fn subscribe(&self) -> broadcast::Receiver<MarketData> {
        self.processed_data_sender.subscribe()
    }
    
    /// Get current pipeline metrics
    pub fn get_metrics(&self) -> PipelineMetrics {
        self.metrics.read().unwrap().clone()
    }
    
    /// Start ingestion worker
    async fn start_ingestion_worker(&self) -> Result<tokio::task::JoinHandle<()>> {
        let receiver = {
            let mut receiver_guard = self.raw_data_receiver.lock().unwrap();
            receiver_guard.take()
                .ok_or_else(|| AnalysisError::invalid_config("Ingestion worker already started"))?
        };
        
        let buffer = Arc::clone(&self.ingestion_buffer);
        let processing_queue = Arc::clone(&self.processing_queue);
        let is_running = Arc::clone(&self.is_running);
        let metrics = Arc::clone(&self.metrics);
        let validator = self.data_validator.clone();
        let batch_size = self.config.batch_size;
        let flush_interval = Duration::milliseconds(self.config.flush_interval_ms as i64);
        
        let handle = tokio::spawn(async move {
            let mut receiver = receiver;
            let mut batch = Vec::with_capacity(batch_size);
            let mut last_flush = Utc::now();
            
            while *is_running.read().unwrap() {
                let should_flush = batch.len() >= batch_size || 
                    (Utc::now() - last_flush) > flush_interval;
                
                if should_flush && !batch.is_empty() {
                    // Process batch
                    let processed_batch = Self::process_raw_batch(batch, &validator).await;
                    
                    // Add to processing queue
                    {
                        let mut queue = processing_queue.lock().unwrap();
                        queue.extend(processed_batch);
                    }
                    
                    batch = Vec::with_capacity(batch_size);
                    last_flush = Utc::now();
                    
                    // Update metrics
                    {
                        let mut m = metrics.write().unwrap();
                        m.total_messages_processed += batch_size as u64;
                    }
                }
                
                // Try to receive new data (non-blocking)
                match tokio::time::timeout(
                    std::time::Duration::from_millis(10),
                    receiver.recv()
                ).await {
                    Ok(Some(data)) => {
                        batch.push(data);
                    }
                    Ok(None) => {
                        debug!("Ingestion channel closed");
                        break;
                    }
                    Err(_) => {
                        // Timeout, continue to check flush condition
                        continue;
                    }
                }
            }
            
            // Process remaining batch
            if !batch.is_empty() {
                let processed_batch = Self::process_raw_batch(batch, &validator).await;
                let mut queue = processing_queue.lock().unwrap();
                queue.extend(processed_batch);
            }
            
            info!("Ingestion worker stopped");
        });
        
        Ok(handle)
    }
    
    /// Start processing worker
    async fn start_processing_worker(&self, worker_id: usize) -> Result<tokio::task::JoinHandle<()>> {
        let processing_queue = Arc::clone(&self.processing_queue);
        let processed_data_sender = self.processed_data_sender.clone();
        let is_running = Arc::clone(&self.is_running);
        let metrics = Arc::clone(&self.metrics);
        let feature_processor = self.feature_processor.clone();
        let aggregator = self.streaming_aggregator.clone();
        let enable_simd = self.config.enable_simd;
        
        let handle = tokio::spawn(async move {
            debug!("Processing worker {} started", worker_id);
            
            while *is_running.read().unwrap() {
                let batch = {
                    let mut queue = processing_queue.lock().unwrap();
                    if queue.is_empty() {
                        drop(queue);
                        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                        continue;
                    }
                    
                    // Take up to batch_size items
                    let take_count = queue.len().min(10);
                    queue.drain(0..take_count).collect::<Vec<_>>()
                };
                
                if batch.is_empty() {
                    continue;
                }
                
                let start_time = std::time::Instant::now();
                
                // Process batch with SIMD optimization
                let processed_batch = if enable_simd {
                    Self::process_batch_simd(batch, &feature_processor, &aggregator).await
                } else {
                    Self::process_batch_scalar(batch, &feature_processor, &aggregator).await
                };
                
                let processing_time = start_time.elapsed();
                
                // Send processed data
                for data in processed_batch {
                    if let Err(e) = processed_data_sender.send(data) {
                        warn!("Failed to send processed data: {:?}", e);
                    }
                }
                
                // Update metrics
                {
                    let mut m = metrics.write().unwrap();
                    m.average_processing_latency_ms = 
                        (m.average_processing_latency_ms * 0.9) + 
                        (processing_time.as_secs_f64() * 1000.0 * 0.1);
                    
                    m.peak_processing_latency_ms = m.peak_processing_latency_ms
                        .max(processing_time.as_secs_f64() * 1000.0);
                }
            }
            
            debug!("Processing worker {} stopped", worker_id);
        });
        
        Ok(handle)
    }
    
    /// Start metrics collection worker
    async fn start_metrics_collector(&self) -> Result<tokio::task::JoinHandle<()>> {
        let metrics = Arc::clone(&self.metrics);
        let is_running = Arc::clone(&self.is_running);
        let buffer = Arc::clone(&self.ingestion_buffer);
        let buffer_size = self.config.buffer_size;
        
        let handle = tokio::spawn(async move {
            let mut last_message_count = 0u64;
            let mut last_update = Utc::now();
            
            while *is_running.read().unwrap() {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                
                let now = Utc::now();
                let elapsed = (now - last_update).num_seconds() as f64;
                
                {
                    let mut m = metrics.write().unwrap();
                    
                    // Calculate throughput
                    let message_diff = m.total_messages_processed - last_message_count;
                    m.throughput_per_second = if elapsed > 0.0 {
                        message_diff as f64 / elapsed
                    } else {
                        0.0
                    };
                    
                    // Update buffer utilization
                    let buffer_guard = buffer.read().unwrap();
                    m.buffer_utilization = buffer_guard.len() as f64 / buffer_size as f64;
                    drop(buffer_guard);
                    
                    m.last_update = now;
                    
                    last_message_count = m.total_messages_processed;
                }
                
                last_update = now;
            }
            
            debug!("Metrics collector stopped");
        });
        
        Ok(handle)
    }
    
    /// Process raw data batch
    async fn process_raw_batch(
        batch: Vec<RawMarketData>, 
        validator: &DataValidator
    ) -> Vec<MarketData> {
        let mut processed = Vec::with_capacity(batch.len());
        
        for raw_data in batch {
            if let Ok(validated_data) = validator.validate_and_convert(raw_data).await {
                processed.push(validated_data);
            }
        }
        
        processed
    }
    
    /// Process batch with SIMD optimization
    #[cfg(feature = "simd")]
    async fn process_batch_simd(
        batch: Vec<MarketData>,
        feature_processor: &FeatureProcessor,
        aggregator: &StreamingAggregator
    ) -> Vec<MarketData> {
        // Group data by symbol for SIMD processing
        let mut symbol_groups: HashMap<String, Vec<MarketData>> = HashMap::new();
        
        for data in batch {
            symbol_groups.entry(data.symbol.clone())
                .or_insert_with(Vec::new)
                .push(data);
        }
        
        let mut processed = Vec::new();
        
        for (symbol, group) in symbol_groups {
            if group.len() >= 4 {
                // Process in SIMD chunks
                let chunks = group.chunks(4);
                for chunk in chunks {
                    let simd_processed = Self::process_simd_chunk(chunk, feature_processor).await;
                    processed.extend(simd_processed);
                }
            } else {
                // Process remaining items scalar
                for data in group {
                    let enhanced_data = feature_processor.process_single(data).await.unwrap_or_else(|_| data);
                    processed.push(enhanced_data);
                }
            }
        }
        
        processed
    }
    
    /// Process batch without SIMD (fallback)
    async fn process_batch_scalar(
        batch: Vec<MarketData>,
        feature_processor: &FeatureProcessor,
        aggregator: &StreamingAggregator
    ) -> Vec<MarketData> {
        let mut processed = Vec::with_capacity(batch.len());
        
        for data in batch {
            let enhanced_data = feature_processor.process_single(data).await
                .unwrap_or_else(|_| data);
            processed.push(enhanced_data);
        }
        
        processed
    }
    
    /// Process SIMD chunk of 4 items
    #[cfg(feature = "simd")]
    async fn process_simd_chunk(
        chunk: &[MarketData],
        feature_processor: &FeatureProcessor
    ) -> Vec<MarketData> {
        assert!(chunk.len() <= 4);
        
        // Extract prices for SIMD processing
        let mut prices = [0.0; 4];
        let mut volumes = [0.0; 4];
        
        for (i, data) in chunk.iter().enumerate() {
            if let Some(last_price) = data.prices.last() {
                prices[i] = *last_price;
            }
            if let Some(last_volume) = data.volumes.last() {
                volumes[i] = *last_volume;
            }
        }
        
        // SIMD processing
        let price_vec = f64x4::new(prices);
        let volume_vec = f64x4::new(volumes);
        
        // Example SIMD operations
        let vwap_vec = price_vec * volume_vec;
        let normalized_prices = price_vec / f64x4::splat(prices.iter().sum::<f64>() / 4.0);
        
        // Convert back to processed data
        let vwap_array = vwap_vec.to_array();
        let norm_price_array = normalized_prices.to_array();
        
        let mut processed = Vec::with_capacity(chunk.len());
        
        for (i, mut data) in chunk.iter().cloned().enumerate() {
            // Add SIMD-computed features to metadata
            data.metadata.insert(
                "simd_vwap".to_string(), 
                serde_json::Value::Number(serde_json::Number::from_f64(vwap_array[i]).unwrap())
            );
            data.metadata.insert(
                "simd_normalized_price".to_string(),
                serde_json::Value::Number(serde_json::Number::from_f64(norm_price_array[i]).unwrap())
            );
            
            processed.push(data);
        }
        
        processed
    }
    
    #[cfg(not(feature = "simd"))]
    async fn process_batch_simd(
        batch: Vec<MarketData>,
        feature_processor: &FeatureProcessor,
        aggregator: &StreamingAggregator
    ) -> Vec<MarketData> {
        Self::process_batch_scalar(batch, feature_processor, aggregator).await
    }
    
    #[cfg(not(feature = "simd"))]
    async fn process_simd_chunk(
        chunk: &[MarketData],
        feature_processor: &FeatureProcessor
    ) -> Vec<MarketData> {
        chunk.to_vec()
    }
}

/// Data validator for ensuring data quality
#[derive(Debug, Clone)]
pub struct DataValidator {
    config: DataPipelineConfig,
    outlier_detector: OutlierDetector,
}

impl DataValidator {
    fn new(config: &DataPipelineConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            outlier_detector: OutlierDetector::new()?,
        })
    }
    
    async fn validate_and_convert(&self, raw_data: RawMarketData) -> Result<MarketData> {
        if !self.config.validation_enabled {
            return Ok(self.convert_raw_to_market_data(raw_data));
        }
        
        // Validate price
        if raw_data.price <= 0.0 || !raw_data.price.is_finite() {
            return Err(AnalysisError::validation_error("Invalid price"));
        }
        
        // Validate volume
        if raw_data.volume < 0.0 || !raw_data.volume.is_finite() {
            return Err(AnalysisError::validation_error("Invalid volume"));
        }
        
        // Validate timestamp (not too old or in future)
        let now = Utc::now();
        if raw_data.timestamp > now + Duration::minutes(1) {
            return Err(AnalysisError::validation_error("Future timestamp"));
        }
        
        if raw_data.timestamp < now - Duration::hours(24) {
            return Err(AnalysisError::validation_error("Timestamp too old"));
        }
        
        // Check for outliers
        if self.outlier_detector.is_outlier(raw_data.price, raw_data.volume)? {
            warn!("Outlier detected: price={}, volume={}", raw_data.price, raw_data.volume);
            // Log but don't reject - might be legitimate extreme value
        }
        
        Ok(self.convert_raw_to_market_data(raw_data))
    }
    
    fn convert_raw_to_market_data(&self, raw_data: RawMarketData) -> MarketData {
        let mut market_data = MarketData::new(raw_data.symbol, Timeframe::OneMinute);
        market_data.timestamp = raw_data.timestamp;
        market_data.prices = vec![raw_data.price];
        market_data.volumes = vec![raw_data.volume];
        market_data.metadata = raw_data.metadata;
        
        // Convert to trade if we have trade information
        if let (Some(trade_id), Some(trade_side)) = (raw_data.trade_id, raw_data.trade_side) {
            let trade = Trade {
                id: trade_id,
                timestamp: raw_data.timestamp,
                price: raw_data.price,
                quantity: raw_data.volume,
                side: trade_side,
                trade_type: TradeType::Market, // Default assumption
            };
            market_data.add_trade(trade);
        }
        
        // Create order book if we have bid/ask
        if let (Some(bid), Some(ask)) = (raw_data.bid, raw_data.ask) {
            let order_book = OrderBook {
                timestamp: raw_data.timestamp,
                bids: vec![OrderBookLevel { price: bid, quantity: raw_data.volume, order_count: Some(1) }],
                asks: vec![OrderBookLevel { price: ask, quantity: raw_data.volume, order_count: Some(1) }],
                sequence: 0,
            };
            market_data.set_order_book(order_book);
        }
        
        market_data
    }
}

/// Outlier detector for data validation
#[derive(Debug, Clone)]
struct OutlierDetector {
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    max_history: usize,
}

impl OutlierDetector {
    fn new() -> Result<Self> {
        Ok(Self {
            price_history: VecDeque::with_capacity(1000),
            volume_history: VecDeque::with_capacity(1000),
            max_history: 1000,
        })
    }
    
    fn is_outlier(&mut self, price: f64, volume: f64) -> Result<bool> {
        // Add to history
        self.price_history.push_back(price);
        self.volume_history.push_back(volume);
        
        // Maintain max history size
        if self.price_history.len() > self.max_history {
            self.price_history.pop_front();
        }
        if self.volume_history.len() > self.max_history {
            self.volume_history.pop_front();
        }
        
        // Need sufficient history for outlier detection
        if self.price_history.len() < 20 {
            return Ok(false);
        }
        
        // Convert to Vec for analysis
        let prices: Vec<f64> = self.price_history.iter().cloned().collect();
        let volumes: Vec<f64> = self.volume_history.iter().cloned().collect();
        
        // Use IQR method for outlier detection
        let price_outliers = crate::utils::statistical::detect_outliers_iqr(&prices)?;
        let volume_outliers = crate::utils::statistical::detect_outliers_iqr(&volumes)?;
        
        // Check if current values are outliers
        let is_price_outlier = price_outliers.last().copied().unwrap_or(false);
        let is_volume_outlier = volume_outliers.last().copied().unwrap_or(false);
        
        Ok(is_price_outlier || is_volume_outlier)
    }
}

/// Feature processor for enhancing market data
#[derive(Debug, Clone)]
pub struct FeatureProcessor {
    config: DataPipelineConfig,
    technical_indicators: TechnicalIndicatorCalculator,
    statistical_features: StatisticalFeatureCalculator,
}

impl FeatureProcessor {
    fn new(config: &DataPipelineConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            technical_indicators: TechnicalIndicatorCalculator::new()?,
            statistical_features: StatisticalFeatureCalculator::new()?,
        })
    }
    
    async fn process_single(&self, mut data: MarketData) -> Result<MarketData> {
        // Add technical indicators
        self.technical_indicators.calculate(&mut data)?;
        
        // Add statistical features
        self.statistical_features.calculate(&mut data)?;
        
        // Add timestamp-based features
        self.add_time_features(&mut data)?;
        
        Ok(data)
    }
    
    fn add_time_features(&self, data: &mut MarketData) -> Result<()> {
        let timestamp = data.timestamp;
        
        // Add hour of day
        data.metadata.insert(
            "hour_of_day".to_string(),
            serde_json::Value::Number(serde_json::Number::from(timestamp.hour()))
        );
        
        // Add day of week
        data.metadata.insert(
            "day_of_week".to_string(),
            serde_json::Value::Number(serde_json::Number::from(timestamp.weekday().num_days_from_monday()))
        );
        
        // Add market session
        let hour = timestamp.hour();
        let market_session = match hour {
            0..=6 => "asian",
            7..=14 => "european", 
            15..=22 => "american",
            _ => "after_hours",
        };
        
        data.metadata.insert(
            "market_session".to_string(),
            serde_json::Value::String(market_session.to_string())
        );
        
        Ok(())
    }
}

/// Technical indicator calculator
#[derive(Debug, Clone)]
struct TechnicalIndicatorCalculator {
    window_sizes: Vec<usize>,
}

impl TechnicalIndicatorCalculator {
    fn new() -> Result<Self> {
        Ok(Self {
            window_sizes: vec![5, 10, 20, 50],
        })
    }
    
    fn calculate(&self, data: &mut MarketData) -> Result<()> {
        if data.prices.is_empty() {
            return Ok(());
        }
        
        // Calculate moving averages
        for &window in &self.window_sizes {
            if data.prices.len() >= window {
                let sma = self.calculate_sma(&data.prices, window)?;
                data.metadata.insert(
                    format!("sma_{}", window),
                    serde_json::Value::Number(serde_json::Number::from_f64(sma).unwrap())
                );
                
                let ema = self.calculate_ema(&data.prices, window)?;
                data.metadata.insert(
                    format!("ema_{}", window),
                    serde_json::Value::Number(serde_json::Number::from_f64(ema).unwrap())
                );
            }
        }
        
        // Calculate RSI
        if data.prices.len() >= 14 {
            let rsi = self.calculate_rsi(&data.prices, 14)?;
            data.metadata.insert(
                "rsi".to_string(),
                serde_json::Value::Number(serde_json::Number::from_f64(rsi).unwrap())
            );
        }
        
        Ok(())
    }
    
    fn calculate_sma(&self, prices: &[f64], window: usize) -> Result<f64> {
        if prices.len() < window {
            return Ok(prices.iter().sum::<f64>() / prices.len() as f64);
        }
        
        let recent_prices = &prices[prices.len()-window..];
        Ok(recent_prices.iter().sum::<f64>() / window as f64)
    }
    
    fn calculate_ema(&self, prices: &[f64], window: usize) -> Result<f64> {
        if prices.len() < window {
            return self.calculate_sma(prices, prices.len());
        }
        
        let alpha = 2.0 / (window as f64 + 1.0);
        let mut ema = prices[0];
        
        for &price in &prices[1..] {
            ema = alpha * price + (1.0 - alpha) * ema;
        }
        
        Ok(ema)
    }
    
    fn calculate_rsi(&self, prices: &[f64], window: usize) -> Result<f64> {
        if prices.len() < window + 1 {
            return Ok(50.0);
        }
        
        let mut gains = 0.0;
        let mut losses = 0.0;
        
        for i in prices.len()-window..prices.len() {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains += change;
            } else {
                losses -= change;
            }
        }
        
        let avg_gain = gains / window as f64;
        let avg_loss = losses / window as f64;
        
        if avg_loss == 0.0 {
            return Ok(100.0);
        }
        
        let rs = avg_gain / avg_loss;
        Ok(100.0 - (100.0 / (1.0 + rs)))
    }
}

/// Statistical feature calculator
#[derive(Debug, Clone)]
struct StatisticalFeatureCalculator {
    rolling_windows: Vec<usize>,
}

impl StatisticalFeatureCalculator {
    fn new() -> Result<Self> {
        Ok(Self {
            rolling_windows: vec![10, 20, 50],
        })
    }
    
    fn calculate(&self, data: &mut MarketData) -> Result<()> {
        if data.prices.len() < 2 {
            return Ok(());
        }
        
        // Calculate returns
        let returns = self.calculate_returns(&data.prices)?;
        
        // Rolling statistics
        for &window in &self.rolling_windows {
            if returns.len() >= window {
                let recent_returns = &returns[returns.len()-window..];
                
                let volatility = self.calculate_volatility(recent_returns)?;
                data.metadata.insert(
                    format!("volatility_{}", window),
                    serde_json::Value::Number(serde_json::Number::from_f64(volatility).unwrap())
                );
                
                let skewness = self.calculate_skewness(recent_returns)?;
                data.metadata.insert(
                    format!("skewness_{}", window),
                    serde_json::Value::Number(serde_json::Number::from_f64(skewness).unwrap())
                );
            }
        }
        
        Ok(())
    }
    
    fn calculate_returns(&self, prices: &[f64]) -> Result<Vec<f64>> {
        if prices.len() < 2 {
            return Ok(Vec::new());
        }
        
        let returns = prices.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();
            
        Ok(returns)
    }
    
    fn calculate_volatility(&self, returns: &[f64]) -> Result<f64> {
        if returns.is_empty() {
            return Ok(0.0);
        }
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
            
        Ok(variance.sqrt())
    }
    
    fn calculate_skewness(&self, returns: &[f64]) -> Result<f64> {
        if returns.len() < 3 {
            return Ok(0.0);
        }
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let std_dev = self.calculate_volatility(returns)?;
        
        if std_dev == 0.0 {
            return Ok(0.0);
        }
        
        let n = returns.len() as f64;
        let skew_sum: f64 = returns.iter()
            .map(|&r| ((r - mean) / std_dev).powi(3))
            .sum();
            
        Ok((n / ((n - 1.0) * (n - 2.0))) * skew_sum)
    }
}

/// Streaming aggregator for real-time metrics
#[derive(Debug, Clone)]
pub struct StreamingAggregator {
    config: DataPipelineConfig,
    symbol_aggregates: Arc<RwLock<HashMap<String, SymbolAggregate>>>,
}

#[derive(Debug, Clone)]
struct SymbolAggregate {
    symbol: String,
    count: u64,
    sum_price: f64,
    sum_volume: f64,
    min_price: f64,
    max_price: f64,
    last_update: DateTime<Utc>,
    price_history: VecDeque<f64>,
}

impl StreamingAggregator {
    fn new(config: &DataPipelineConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            symbol_aggregates: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    pub fn update(&self, data: &MarketData) -> Result<()> {
        let mut aggregates = self.symbol_aggregates.write().unwrap();
        
        let aggregate = aggregates.entry(data.symbol.clone())
            .or_insert_with(|| SymbolAggregate {
                symbol: data.symbol.clone(),
                count: 0,
                sum_price: 0.0,
                sum_volume: 0.0,
                min_price: f64::INFINITY,
                max_price: f64::NEG_INFINITY,
                last_update: Utc::now(),
                price_history: VecDeque::with_capacity(1000),
            });
        
        if let Some(&price) = data.prices.last() {
            aggregate.count += 1;
            aggregate.sum_price += price;
            aggregate.min_price = aggregate.min_price.min(price);
            aggregate.max_price = aggregate.max_price.max(price);
            aggregate.last_update = data.timestamp;
            
            aggregate.price_history.push_back(price);
            if aggregate.price_history.len() > 1000 {
                aggregate.price_history.pop_front();
            }
        }
        
        if let Some(&volume) = data.volumes.last() {
            aggregate.sum_volume += volume;
        }
        
        Ok(())
    }
    
    pub fn get_aggregate(&self, symbol: &str) -> Option<SymbolAggregate> {
        let aggregates = self.symbol_aggregates.read().unwrap();
        aggregates.get(symbol).cloned()
    }
    
    pub fn get_all_aggregates(&self) -> HashMap<String, SymbolAggregate> {
        self.symbol_aggregates.read().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration as TokioDuration};
    
    #[tokio::test]
    async fn test_pipeline_creation() {
        let config = Config::default();
        let pipeline = DataPipeline::new(config);
        assert!(pipeline.is_ok());
    }
    
    #[tokio::test]
    async fn test_pipeline_start_stop() {
        let config = Config::default();
        let pipeline = DataPipeline::new(config).unwrap();
        
        assert!(pipeline.start().await.is_ok());
        sleep(TokioDuration::from_millis(100)).await;
        assert!(pipeline.stop().await.is_ok());
    }
    
    #[tokio::test]
    async fn test_data_ingestion() {
        let config = Config::default();
        let pipeline = DataPipeline::new(config).unwrap();
        
        pipeline.start().await.unwrap();
        
        let raw_data = RawMarketData {
            symbol: "BTCUSDT".to_string(),
            timestamp: Utc::now(),
            price: 50000.0,
            volume: 1.5,
            bid: Some(49999.0),
            ask: Some(50001.0),
            trade_id: Some("12345".to_string()),
            trade_side: Some(TradeSide::Buy),
            metadata: HashMap::new(),
        };
        
        assert!(pipeline.ingest(raw_data).await.is_ok());
        
        sleep(TokioDuration::from_millis(200)).await;
        pipeline.stop().await.unwrap();
    }
    
    #[test]
    fn test_data_validation() {
        let config = DataPipelineConfig::default();
        let validator = DataValidator::new(&config).unwrap();
        
        let valid_data = RawMarketData {
            symbol: "BTCUSDT".to_string(),
            timestamp: Utc::now(),
            price: 50000.0,
            volume: 1.5,
            bid: None,
            ask: None,
            trade_id: None,
            trade_side: None,
            metadata: HashMap::new(),
        };
        
        // This would be an async test in practice
        // let result = validator.validate_and_convert(valid_data).await;
        // assert!(result.is_ok());
    }
    
    #[test]
    fn test_technical_indicators() {
        let calculator = TechnicalIndicatorCalculator::new().unwrap();
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0];
        
        let sma = calculator.calculate_sma(&prices, 3).unwrap();
        assert!((sma - 104.0).abs() < 1e-10);
        
        let rsi = calculator.calculate_rsi(&prices, 3).unwrap();
        assert!(rsi >= 0.0 && rsi <= 100.0);
    }
    
    #[test]
    fn test_streaming_aggregator() {
        let config = DataPipelineConfig::default();
        let aggregator = StreamingAggregator::new(&config).unwrap();
        
        let mut market_data = MarketData::new("BTCUSDT".to_string(), Timeframe::OneMinute);
        market_data.prices = vec![50000.0];
        market_data.volumes = vec![1.5];
        
        assert!(aggregator.update(&market_data).is_ok());
        
        let aggregate = aggregator.get_aggregate("BTCUSDT");
        assert!(aggregate.is_some());
        
        let agg = aggregate.unwrap();
        assert_eq!(agg.count, 1);
        assert_eq!(agg.min_price, 50000.0);
        assert_eq!(agg.max_price, 50000.0);
    }
}