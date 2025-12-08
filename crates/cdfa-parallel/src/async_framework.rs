//! Async processing framework for CDFA using Tokio
//!
//! Provides high-performance async/await patterns for streaming signal processing
//! with backpressure handling and efficient resource utilization.

use async_trait::async_trait;
use futures::{
    channel::mpsc,
    stream::{Stream, StreamExt},
    SinkExt,
};
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
use tokio::sync::{Semaphore, RwLock as AsyncRwLock};
use tokio::time::{interval, sleep};

use cdfa_core::error::Result;
use cdfa_core::traits::{CognitiveDiversityAnalyzer, SignalProcessor};
use cdfa_core::types::{AnalysisResult, Signal};

use crate::lock_free::{LockFreeResultAggregator, LockFreeSignalBuffer};

/// Async signal stream processor
///
/// Processes signals in a streaming fashion with configurable parallelism
/// and backpressure handling.
pub struct AsyncSignalProcessor {
    /// Bounded channel for incoming signals
    receiver: mpsc::Receiver<Signal>,
    
    /// Signal processors
    processors: Vec<Arc<dyn SignalProcessor>>,
    
    /// Processing semaphore for concurrency control
    semaphore: Arc<Semaphore>,
    
    /// Buffer for processed signals
    buffer: VecDeque<Signal>,
    
    /// Maximum buffer size before applying backpressure
    max_buffer_size: usize,
    
    /// Processing statistics
    stats: Arc<AsyncRwLock<ProcessingStats>>,
}

/// Processing statistics
#[derive(Debug, Default, Clone)]
pub struct ProcessingStats {
    pub signals_processed: u64,
    pub total_latency_ns: u64,
    pub backpressure_events: u64,
    pub errors: u64,
}

impl AsyncSignalProcessor {
    /// Creates a new async signal processor
    pub fn new(
        receiver: mpsc::Receiver<Signal>,
        processors: Vec<Arc<dyn SignalProcessor>>,
        max_concurrency: usize,
        max_buffer_size: usize,
    ) -> Self {
        Self {
            receiver,
            processors,
            semaphore: Arc::new(Semaphore::new(max_concurrency)),
            buffer: VecDeque::with_capacity(max_buffer_size),
            max_buffer_size,
            stats: Arc::new(AsyncRwLock::new(ProcessingStats::default())),
        }
    }
    
    /// Processes signals asynchronously
    async fn process_signal(&self, signal: Signal) -> Result<Signal> {
        let start = Instant::now();
        let _permit = self.semaphore.acquire().await.unwrap();
        
        // Apply all processors in sequence
        let mut processed = signal;
        for processor in &self.processors {
            processed = processor.process(&processed)?;
        }
        
        // Update stats
        let latency_ns = start.elapsed().as_nanos() as u64;
        let mut stats = self.stats.write().await;
        stats.signals_processed += 1;
        stats.total_latency_ns += latency_ns;
        
        Ok(processed)
    }
    
    /// Gets current processing statistics
    pub async fn stats(&self) -> ProcessingStats {
        self.stats.read().await.clone()
    }
}

impl Stream for AsyncSignalProcessor {
    type Item = Result<Signal>;
    
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Check if we have buffered signals
        if let Some(signal) = self.buffer.pop_front() {
            return Poll::Ready(Some(Ok(signal)));
        }
        
        // Apply backpressure if buffer is full
        if self.buffer.len() >= self.max_buffer_size {
            let mut stats = futures::executor::block_on(self.stats.write());
            stats.backpressure_events += 1;
            return Poll::Pending;
        }
        
        // Try to receive new signals
        match Pin::new(&mut self.receiver).poll_next(cx) {
            Poll::Ready(Some(signal)) => {
                // Process signal asynchronously
                let processed = futures::executor::block_on(self.process_signal(signal));
                match processed {
                    Ok(sig) => Poll::Ready(Some(Ok(sig))),
                    Err(e) => {
                        let mut stats = futures::executor::block_on(self.stats.write());
                        stats.errors += 1;
                        Poll::Ready(Some(Err(e)))
                    }
                }
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Async diversity analyzer with parallel execution
///
/// Runs multiple analyzers concurrently and aggregates results
pub struct AsyncDiversityAnalyzer {
    /// Cognitive diversity analyzers
    analyzers: Vec<Arc<dyn CognitiveDiversityAnalyzer>>,
    
    /// Result aggregator
    aggregator: Arc<LockFreeResultAggregator>,
    
    /// Concurrency limiter
    semaphore: Arc<Semaphore>,
    
    /// Analysis timeout
    timeout: Duration,
}

impl AsyncDiversityAnalyzer {
    /// Creates a new async diversity analyzer
    pub fn new(
        analyzers: Vec<Arc<dyn CognitiveDiversityAnalyzer>>,
        max_concurrency: usize,
        timeout: Duration,
    ) -> Self {
        Self {
            analyzers,
            aggregator: Arc::new(LockFreeResultAggregator::new()),
            semaphore: Arc::new(Semaphore::new(max_concurrency)),
            timeout,
        }
    }
    
    /// Analyzes signals using all analyzers in parallel
    pub async fn analyze_parallel(&self, signals: &[Signal]) -> Result<Vec<AnalysisResult>> {
        let mut handles = Vec::with_capacity(self.analyzers.len());
        
        // Spawn analysis tasks
        for analyzer in &self.analyzers {
            let analyzer = Arc::clone(analyzer);
            let signals = signals.to_vec();
            let aggregator = Arc::clone(&self.aggregator);
            let semaphore = Arc::clone(&self.semaphore);
            let timeout = self.timeout;
            
            let handle = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                
                // Run analysis with timeout
                match tokio::time::timeout(timeout, async {
                    analyzer.analyze(&signals)
                }).await {
                    Ok(Ok(result)) => {
                        aggregator.add_result(result);
                        Ok(())
                    }
                    Ok(Err(e)) => Err(e),
                    Err(_) => Err(cdfa_core::error::Error::Timeout),
                }
            });
            
            handles.push(handle);
        }
        
        // Wait for all analyses to complete
        for handle in handles {
            handle.await.map_err(|_| cdfa_core::error::Error::Internal)?;
        }
        
        // Collect all results
        Ok(self.aggregator.collect_results())
    }
    
    /// Gets analysis statistics
    pub fn stats(&self) -> crate::lock_free::AggregatorStats {
        self.aggregator.stats()
    }
}

/// Streaming pipeline for continuous signal processing
///
/// Provides a high-level interface for building async processing pipelines
pub struct StreamingPipeline {
    /// Signal source
    source: mpsc::Receiver<Signal>,
    
    /// Processing stages
    stages: Vec<ProcessingStage>,
    
    /// Output sink
    sink: mpsc::Sender<AnalysisResult>,
    
    /// Pipeline configuration
    config: PipelineConfig,
    
    /// Lock-free signal buffer for buffering
    buffer: Arc<LockFreeSignalBuffer>,
}

/// Processing stage in the pipeline
struct ProcessingStage {
    name: String,
    processor: Arc<dyn SignalProcessor>,
    analyzer: Arc<dyn CognitiveDiversityAnalyzer>,
}

/// Pipeline configuration
#[derive(Clone)]
pub struct PipelineConfig {
    /// Maximum signals to buffer
    pub buffer_size: usize,
    
    /// Processing batch size
    pub batch_size: usize,
    
    /// Batch timeout
    pub batch_timeout: Duration,
    
    /// Maximum concurrent batches
    pub max_concurrency: usize,
    
    /// Enable adaptive batching
    pub adaptive_batching: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            buffer_size: 10_000,
            batch_size: 64,
            batch_timeout: Duration::from_millis(10),
            max_concurrency: 4,
            adaptive_batching: true,
        }
    }
}

impl StreamingPipeline {
    /// Creates a new streaming pipeline
    pub fn new(
        source: mpsc::Receiver<Signal>,
        sink: mpsc::Sender<AnalysisResult>,
        config: PipelineConfig,
    ) -> Self {
        Self {
            source,
            stages: Vec::new(),
            sink,
            buffer: Arc::new(LockFreeSignalBuffer::new(config.buffer_size)),
            config,
        }
    }
    
    /// Adds a processing stage to the pipeline
    pub fn add_stage(
        &mut self,
        name: impl Into<String>,
        processor: Arc<dyn SignalProcessor>,
        analyzer: Arc<dyn CognitiveDiversityAnalyzer>,
    ) {
        self.stages.push(ProcessingStage {
            name: name.into(),
            processor,
            analyzer,
        });
    }
    
    /// Runs the pipeline
    pub async fn run(mut self) -> Result<()> {
        // Spawn buffer filler task
        let buffer = Arc::clone(&self.buffer);
        let mut source = self.source;
        
        tokio::spawn(async move {
            while let Some(signal) = source.next().await {
                // Try to push to buffer, drop on overflow
                let _ = buffer.push(signal);
            }
        });
        
        // Process batches
        let mut batch_interval = interval(self.config.batch_timeout);
        let mut current_batch = Vec::with_capacity(self.config.batch_size);
        
        loop {
            tokio::select! {
                _ = batch_interval.tick() => {
                    // Process current batch if not empty
                    if !current_batch.is_empty() {
                        self.process_batch(current_batch).await?;
                        current_batch = Vec::with_capacity(self.config.batch_size);
                    }
                }
                
                // Try to fill batch from buffer
                _ = sleep(Duration::from_micros(100)) => {
                    while current_batch.len() < self.config.batch_size {
                        if let Some(signal) = self.buffer.try_pop() {
                            current_batch.push(signal);
                        } else {
                            break;
                        }
                    }
                    
                    // Process full batch
                    if current_batch.len() >= self.config.batch_size {
                        self.process_batch(current_batch).await?;
                        current_batch = Vec::with_capacity(self.config.batch_size);
                    }
                }
            }
        }
    }
    
    /// Processes a batch of signals through all stages
    async fn process_batch(&mut self, mut signals: Vec<Signal>) -> Result<()> {
        for stage in &self.stages {
            // Process signals
            let mut processed = Vec::with_capacity(signals.len());
            for signal in signals {
                processed.push(stage.processor.process(&signal)?);
            }
            signals = processed;
            
            // Analyze processed signals
            let result = stage.analyzer.analyze(&signals)?;
            
            // Send result to sink
            self.sink.send(result).await
                .map_err(|_| cdfa_core::error::Error::ChannelClosed)?;
        }
        
        Ok(())
    }
}

/// Backpressure controller for adaptive flow control
pub struct BackpressureController {
    /// Target latency in nanoseconds
    target_latency_ns: u64,
    
    /// Current batch size
    batch_size: usize,
    
    /// Minimum batch size
    min_batch_size: usize,
    
    /// Maximum batch size
    max_batch_size: usize,
    
    /// Adjustment factor
    adjustment_factor: f64,
    
    /// Recent latencies for moving average
    recent_latencies: VecDeque<u64>,
    
    /// Window size for moving average
    window_size: usize,
}

impl BackpressureController {
    /// Creates a new backpressure controller
    pub fn new(
        target_latency_ns: u64,
        initial_batch_size: usize,
        min_batch_size: usize,
        max_batch_size: usize,
    ) -> Self {
        Self {
            target_latency_ns,
            batch_size: initial_batch_size,
            min_batch_size,
            max_batch_size,
            adjustment_factor: 0.1,
            recent_latencies: VecDeque::with_capacity(100),
            window_size: 100,
        }
    }
    
    /// Updates the controller with a new latency measurement
    pub fn update(&mut self, latency_ns: u64) {
        // Add to recent latencies
        self.recent_latencies.push_back(latency_ns);
        if self.recent_latencies.len() > self.window_size {
            self.recent_latencies.pop_front();
        }
        
        // Calculate moving average
        if self.recent_latencies.len() >= 10 {
            let avg_latency = self.recent_latencies.iter().sum::<u64>() 
                / self.recent_latencies.len() as u64;
            
            // Adjust batch size based on latency
            if avg_latency > self.target_latency_ns {
                // Reduce batch size to decrease latency
                let reduction = (self.batch_size as f64 * self.adjustment_factor) as usize;
                self.batch_size = (self.batch_size - reduction).max(self.min_batch_size);
            } else if avg_latency < self.target_latency_ns * 8 / 10 {
                // Increase batch size to improve throughput
                let increase = (self.batch_size as f64 * self.adjustment_factor) as usize;
                self.batch_size = (self.batch_size + increase).min(self.max_batch_size);
            }
        }
    }
    
    /// Gets the current recommended batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
    
    /// Gets the average latency
    pub fn avg_latency_ns(&self) -> Option<u64> {
        if self.recent_latencies.is_empty() {
            None
        } else {
            Some(self.recent_latencies.iter().sum::<u64>() / self.recent_latencies.len() as u64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::channel::mpsc;
    use cdfa_core::types::SignalId;
    
    #[tokio::test]
    async fn test_backpressure_controller() {
        let mut controller = BackpressureController::new(
            1000, // 1 microsecond target
            64,   // initial batch size
            16,   // min batch size
            256,  // max batch size
        );
        
        // Test adjustment for high latency
        for _ in 0..20 {
            controller.update(2000); // 2 microseconds
        }
        assert!(controller.batch_size() < 64);
        
        // Test adjustment for low latency
        for _ in 0..20 {
            controller.update(500); // 0.5 microseconds
        }
        assert!(controller.batch_size() > 32);
    }
    
    #[tokio::test]
    async fn test_async_diversity_analyzer() {
        // Create mock analyzer
        struct MockAnalyzer;
        
        impl CognitiveDiversityAnalyzer for MockAnalyzer {
            type Config = ();
            
            fn analyze(&self, _signals: &[Signal]) -> Result<AnalysisResult> {
                Ok(AnalysisResult::new("mock".to_string(), 0.75, 0.9))
            }
            
            fn diversity_metric_ids(&self) -> &[&'static str] {
                &["mock_metric"]
            }
            
            fn config(&self) -> &Self::Config {
                ()
            }
            
            fn update_config(&mut self, _config: Self::Config) -> Result<()> {
                Ok(())
            }
            
            fn analyzer_id(&self) -> &'static str {
                "mock_analyzer"
            }
        }
        
        // Test parallel analysis
        let analyzers: Vec<Arc<dyn CognitiveDiversityAnalyzer>> = vec![
            Arc::new(MockAnalyzer),
            Arc::new(MockAnalyzer),
            Arc::new(MockAnalyzer),
        ];
        
        let analyzer = AsyncDiversityAnalyzer::new(
            analyzers,
            3,
            Duration::from_secs(1),
        );
        
        let signals = vec![
            Signal::new(SignalId(1), 1000, vec![1.0, 2.0, 3.0]),
            Signal::new(SignalId(2), 2000, vec![4.0, 5.0, 6.0]),
        ];
        
        let results = analyzer.analyze_parallel(&signals).await.unwrap();
        assert_eq!(results.len(), 3);
        
        let stats = analyzer.stats();
        assert_eq!(stats.count, 3);
    }
}