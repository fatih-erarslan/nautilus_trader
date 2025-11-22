//! High-performance optimizations for LMSR calculations
//! 
//! This module provides memory management, parallel processing, and
//! cache optimization strategies for high-frequency trading systems.

use crate::errors::{LMSRError, Result};
use crate::lmsr::{LMSRCalculator, LMSRMarketMaker};
use crate::market::Market;

#[cfg(feature = "simd")]
use crate::simd::{SIMDLMSRCalculator, LockFreeMarketState, AdaptiveSIMDExecutor};

use rayon::prelude::*;
use std::sync::Arc;
use parking_lot::RwLock;
use crossbeam::channel;
use aligned_vec::AVec;
use std::time::{Duration, Instant};

/// High-frequency trading optimized market maker
pub struct HFTMarketMaker {
    calculator: Arc<LMSRCalculator>,
    #[cfg(feature = "simd")]
    simd_calculator: Option<Arc<SIMDLMSRCalculator>>,
    #[cfg(feature = "simd")]
    lock_free_state: Option<Arc<LockFreeMarketState>>,
    thread_pool: rayon::ThreadPool,
    trade_queue: channel::Receiver<TradeRequest>,
    trade_sender: channel::Sender<TradeRequest>,
    performance_monitor: Arc<PerformanceMonitor>,
}

/// Trade request for high-frequency processing
#[derive(Debug, Clone)]
pub struct TradeRequest {
    pub trader_id: String,
    pub quantities: Vec<f64>,
    pub timestamp: Instant,
    pub response_sender: channel::Sender<TradeResult>,
}

/// Trade result with performance metrics
#[derive(Debug, Clone)]
pub struct TradeResult {
    pub cost: f64,
    pub new_prices: Vec<f64>,
    pub execution_time: Duration,
    pub trade_id: String,
}

impl HFTMarketMaker {
    /// Create new high-frequency trading market maker
    pub fn new(
        num_outcomes: usize,
        liquidity_parameter: f64,
        num_threads: Option<usize>,
    ) -> Result<Self> {
        let calculator = Arc::new(LMSRCalculator::new(num_outcomes, liquidity_parameter)?);
        
        // Create optimized thread pool for HFT
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads.unwrap_or_else(|| num_cpus::get()))
            .thread_name(|index| format!("hft-worker-{}", index))
            .stack_size(4 * 1024 * 1024) // 4MB stack
            .build()
            .map_err(|e| LMSRError::invalid_market(format!("Thread pool creation failed: {}", e)))?;

        let (trade_sender, trade_queue) = channel::unbounded();
        let performance_monitor = Arc::new(PerformanceMonitor::new());

        #[cfg(feature = "simd")]
        let simd_calculator = Some(Arc::new(SIMDLMSRCalculator::new(num_outcomes, liquidity_parameter)?));
        #[cfg(not(feature = "simd"))]
        let simd_calculator = None;

        #[cfg(feature = "simd")]
        let lock_free_state = Some(Arc::new(LockFreeMarketState::new(num_outcomes, liquidity_parameter)?));
        #[cfg(not(feature = "simd"))]
        let lock_free_state = None;

        Ok(Self {
            calculator,
            simd_calculator,
            lock_free_state,
            thread_pool,
            trade_queue,
            trade_sender,
            performance_monitor,
        })
    }

    /// Process trades with sub-microsecond latency target
    pub fn start_hft_processing(&self) -> Result<()> {
        let trade_queue = self.trade_queue.clone();
        let performance_monitor = Arc::clone(&self.performance_monitor);
        
        #[cfg(feature = "simd")]
        let lock_free_state = self.lock_free_state.clone();
        #[cfg(not(feature = "simd"))]
        let lock_free_state: Option<Arc<()>> = None;

        self.thread_pool.spawn(move || {
            while let Ok(trade_request) = trade_queue.recv() {
                let start_time = Instant::now();
                
                let result = Self::process_single_trade(
                    &trade_request,
                    &lock_free_state,
                    &performance_monitor,
                );
                
                let execution_time = start_time.elapsed();
                performance_monitor.record_execution(execution_time);
                
                match result {
                    Ok(mut trade_result) => {
                        trade_result.execution_time = execution_time;
                        let _ = trade_request.response_sender.send(trade_result);
                    }
                    Err(e) => {
                        // Log error and continue processing
                        eprintln!("Trade processing error: {:?}", e);
                    }
                }
            }
        });

        Ok(())
    }

    /// Process single trade with optimal path selection
    #[cfg(feature = "simd")]
    fn process_single_trade(
        trade_request: &TradeRequest,
        lock_free_state: &Option<Arc<LockFreeMarketState>>,
        performance_monitor: &Arc<PerformanceMonitor>,
    ) -> Result<TradeResult> {
        if let Some(ref state) = lock_free_state {
            // Use lock-free SIMD path for maximum performance
            let cost = state.execute_trade_lockfree(&trade_request.quantities)?;
            let new_prices = state.get_prices_lockfree()?;
            
            Ok(TradeResult {
                cost,
                new_prices,
                execution_time: Duration::new(0, 0), // Will be set by caller
                trade_id: format!("hft_{}", trade_request.timestamp.elapsed().as_nanos()),
            })
        } else {
            Err(LMSRError::invalid_market("Lock-free state not available".to_string()))
        }
    }

    #[cfg(not(feature = "simd"))]
    fn process_single_trade(
        trade_request: &TradeRequest,
        _lock_free_state: &Option<Arc<()>>,
        _performance_monitor: &Arc<PerformanceMonitor>,
    ) -> Result<TradeResult> {
        // Fallback implementation without SIMD
        Err(LMSRError::invalid_market("SIMD features not available".to_string()))
    }

    /// Submit trade for processing
    pub fn submit_trade(&self, trader_id: String, quantities: Vec<f64>) -> Result<channel::Receiver<TradeResult>> {
        let (response_sender, response_receiver) = channel::bounded(1);
        
        let trade_request = TradeRequest {
            trader_id,
            quantities,
            timestamp: Instant::now(),
            response_sender,
        };
        
        self.trade_sender.send(trade_request)
            .map_err(|_| LMSRError::invalid_market("Trade queue full".to_string()))?;
        
        Ok(response_receiver)
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> HFTPerformanceStats {
        let monitor_stats = self.performance_monitor.get_stats();
        
        HFTPerformanceStats {
            thread_count: self.thread_pool.current_num_threads(),
            average_execution_time: monitor_stats.average_execution_time,
            p99_execution_time: monitor_stats.p99_execution_time,
            trades_per_second: monitor_stats.trades_per_second,
            queue_depth: self.trade_queue.len(),
            simd_enabled: self.simd_calculator.is_some(),
        }
    }
}

/// Performance monitoring for HFT systems
pub struct PerformanceMonitor {
    execution_times: RwLock<Vec<Duration>>,
    trade_count: std::sync::atomic::AtomicU64,
    start_time: Instant,
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new() -> Self {
        Self {
            execution_times: RwLock::new(Vec::new()),
            trade_count: std::sync::atomic::AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    /// Record trade execution time
    pub fn record_execution(&self, duration: Duration) {
        self.trade_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        let mut times = self.execution_times.write();
        times.push(duration);
        
        // Keep only recent measurements for memory efficiency
        if times.len() > 10000 {
            times.drain(0..5000);
        }
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> PerformanceStats {
        let times = self.execution_times.read();
        let trade_count = self.trade_count.load(std::sync::atomic::Ordering::Relaxed);
        let uptime = self.start_time.elapsed();
        
        if times.is_empty() {
            return PerformanceStats {
                average_execution_time: Duration::new(0, 0),
                p99_execution_time: Duration::new(0, 0),
                trades_per_second: 0.0,
            };
        }

        let mut sorted_times = times.clone();
        sorted_times.sort();
        
        let average = sorted_times.iter().sum::<Duration>() / sorted_times.len() as u32;
        let p99_index = (sorted_times.len() as f64 * 0.99) as usize;
        let p99 = sorted_times[p99_index.min(sorted_times.len() - 1)];
        
        let trades_per_second = if uptime.as_secs() > 0 {
            trade_count as f64 / uptime.as_secs_f64()
        } else {
            0.0
        };

        PerformanceStats {
            average_execution_time: average,
            p99_execution_time: p99,
            trades_per_second,
        }
    }
}

/// Performance statistics
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub average_execution_time: Duration,
    pub p99_execution_time: Duration,
    pub trades_per_second: f64,
}

/// HFT-specific performance statistics
#[derive(Debug, Clone)]
pub struct HFTPerformanceStats {
    pub thread_count: usize,
    pub average_execution_time: Duration,
    pub p99_execution_time: Duration,
    pub trades_per_second: f64,
    pub queue_depth: usize,
    pub simd_enabled: bool,
}

/// Batch price calculator for large-scale operations
pub struct BatchPriceCalculator {
    calculator: Arc<LMSRCalculator>,
    #[cfg(feature = "simd")]
    simd_calculator: Option<Arc<SIMDLMSRCalculator>>,
    thread_pool: rayon::ThreadPool,
}

impl BatchPriceCalculator {
    /// Create new batch price calculator
    pub fn new(num_outcomes: usize, liquidity_parameter: f64) -> Result<Self> {
        let calculator = Arc::new(LMSRCalculator::new(num_outcomes, liquidity_parameter)?);
        
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus::get())
            .build()
            .map_err(|e| LMSRError::invalid_market(format!("Thread pool creation failed: {}", e)))?;

        #[cfg(feature = "simd")]
        let simd_calculator = Some(Arc::new(SIMDLMSRCalculator::new(num_outcomes, liquidity_parameter)?));
        #[cfg(not(feature = "simd"))]
        let simd_calculator = None;

        Ok(Self {
            calculator,
            simd_calculator,
            thread_pool,
        })
    }

    /// Calculate prices for large batch of quantity vectors
    pub fn calculate_batch_prices(&self, quantities_batch: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if quantities_batch.is_empty() {
            return Ok(Vec::new());
        }

        let total_calculations: usize = quantities_batch.len();
        
        if total_calculations > 10000 {
            self.calculate_large_batch(quantities_batch)
        } else if total_calculations > 100 {
            self.calculate_medium_batch(quantities_batch)
        } else {
            self.calculate_small_batch(quantities_batch)
        }
    }

    /// Large batch processing with memory optimization
    fn calculate_large_batch(&self, quantities_batch: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let chunk_size = 1000;
        let mut results = Vec::with_capacity(quantities_batch.len());

        for chunk in quantities_batch.chunks(chunk_size) {
            #[cfg(feature = "simd")]
            let chunk_results = if let Some(ref simd_calc) = self.simd_calculator {
                simd_calc.batch_marginal_prices_simd(chunk)?
            } else {
                self.thread_pool.install(|| {
                    chunk.par_iter()
                        .map(|quantities| self.calculator.all_marginal_prices(quantities))
                        .collect::<Result<Vec<Vec<f64>>>>()
                })?
            };

            #[cfg(not(feature = "simd"))]
            let chunk_results = self.thread_pool.install(|| {
                chunk.par_iter()
                    .map(|quantities| self.calculator.all_marginal_prices(quantities))
                    .collect::<Result<Vec<Vec<f64>>>>()
            })?;

            results.extend(chunk_results);
        }

        Ok(results)
    }

    /// Medium batch processing
    fn calculate_medium_batch(&self, quantities_batch: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        #[cfg(feature = "simd")]
        if let Some(ref simd_calc) = self.simd_calculator {
            return simd_calc.batch_marginal_prices_simd(quantities_batch);
        }

        self.thread_pool.install(|| {
            quantities_batch.par_iter()
                .map(|quantities| self.calculator.all_marginal_prices(quantities))
                .collect()
        })
    }

    /// Small batch processing with minimal overhead
    fn calculate_small_batch(&self, quantities_batch: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        quantities_batch.iter()
            .map(|quantities| self.calculator.all_marginal_prices(quantities))
            .collect()
    }
}

/// Memory-efficient streaming processor for continuous data
pub struct StreamingProcessor {
    calculator: Arc<LMSRCalculator>,
    buffer: RwLock<Vec<Vec<f64>>>,
    buffer_size: usize,
    processed_count: std::sync::atomic::AtomicU64,
}

impl StreamingProcessor {
    /// Create new streaming processor
    pub fn new(calculator: LMSRCalculator, buffer_size: usize) -> Self {
        Self {
            calculator: Arc::new(calculator),
            buffer: RwLock::new(Vec::with_capacity(buffer_size)),
            buffer_size,
            processed_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Add quantities to processing buffer
    pub fn add_quantities(&self, quantities: Vec<f64>) -> Result<Option<Vec<Vec<f64>>>> {
        let mut buffer = self.buffer.write();
        buffer.push(quantities);
        
        if buffer.len() >= self.buffer_size {
            // Process buffer and return results
            let quantities_batch = buffer.drain(..).collect::<Vec<_>>();
            drop(buffer); // Release lock early
            
            let results = quantities_batch.iter()
                .map(|q| self.calculator.all_marginal_prices(q))
                .collect::<Result<Vec<Vec<f64>>>>()?;
            
            self.processed_count.fetch_add(quantities_batch.len() as u64, std::sync::atomic::Ordering::Relaxed);
            Ok(Some(results))
        } else {
            Ok(None)
        }
    }

    /// Flush remaining buffer
    pub fn flush(&self) -> Result<Vec<Vec<f64>>> {
        let mut buffer = self.buffer.write();
        if buffer.is_empty() {
            return Ok(Vec::new());
        }
        
        let quantities_batch = buffer.drain(..).collect::<Vec<_>>();
        drop(buffer);
        
        let results = quantities_batch.iter()
            .map(|q| self.calculator.all_marginal_prices(q))
            .collect::<Result<Vec<Vec<f64>>>>()?;
        
        self.processed_count.fetch_add(quantities_batch.len() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(results)
    }

    /// Get processed count
    pub fn get_processed_count(&self) -> u64 {
        self.processed_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Adaptive optimization engine for dynamic parameter tuning
pub struct AdaptiveOptimizer {
    performance_history: RwLock<Vec<(usize, Duration)>>, // (thread_count, execution_time)
    optimal_thread_count: std::sync::atomic::AtomicUsize,
    optimal_batch_size: std::sync::atomic::AtomicUsize,
    last_optimization: RwLock<Instant>,
}

impl AdaptiveOptimizer {
    /// Create new adaptive optimizer
    pub fn new() -> Self {
        Self {
            performance_history: RwLock::new(Vec::new()),
            optimal_thread_count: std::sync::atomic::AtomicUsize::new(num_cpus::get()),
            optimal_batch_size: std::sync::atomic::AtomicUsize::new(1000),
            last_optimization: RwLock::new(Instant::now()),
        }
    }

    /// Record performance data point
    pub fn record_performance(&self, thread_count: usize, execution_time: Duration) {
        let mut history = self.performance_history.write();
        history.push((thread_count, execution_time));
        
        // Keep only recent data points
        if history.len() > 100 {
            history.drain(0..50);
        }
    }

    /// Optimize parameters based on performance history
    pub fn optimize_parameters(&self) {
        let mut last_opt = self.last_optimization.write();
        if last_opt.elapsed() < Duration::from_secs(30) {
            return; // Don't optimize too frequently
        }
        *last_opt = Instant::now();
        drop(last_opt);

        let history = self.performance_history.read();
        if history.len() < 10 {
            return;
        }

        // Find optimal thread count
        let mut thread_performance: std::collections::HashMap<usize, Vec<Duration>> = std::collections::HashMap::new();
        
        for &(thread_count, duration) in history.iter() {
            thread_performance.entry(thread_count).or_insert_with(Vec::new).push(duration);
        }

        if let Some(optimal_threads) = thread_performance.iter()
            .min_by_key(|(_, durations)| {
                let avg = durations.iter().sum::<Duration>() / durations.len() as u32;
                avg
            })
            .map(|(threads, _)| *threads)
        {
            self.optimal_thread_count.store(optimal_threads, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Get optimal thread count
    pub fn get_optimal_thread_count(&self) -> usize {
        self.optimal_thread_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get optimal batch size
    pub fn get_optimal_batch_size(&self) -> usize {
        self.optimal_batch_size.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hft_market_maker() {
        let hft_mm = HFTMarketMaker::new(2, 1000.0, Some(2)).unwrap();
        let stats = hft_mm.get_performance_stats();
        assert_eq!(stats.thread_count, 2);
    }

    #[test]
    fn test_performance_monitor() {
        let monitor = PerformanceMonitor::new();
        monitor.record_execution(Duration::from_micros(100));
        monitor.record_execution(Duration::from_micros(150));
        
        let stats = monitor.get_stats();
        assert!(stats.average_execution_time > Duration::new(0, 0));
    }

    #[test]
    fn test_batch_price_calculator() {
        let calc = BatchPriceCalculator::new(3, 100.0).unwrap();
        let quantities_batch = vec![
            vec![10.0, 20.0, 30.0],
            vec![5.0, 15.0, 25.0],
        ];
        
        let results = calc.calculate_batch_prices(&quantities_batch).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 3);
        assert_eq!(results[1].len(), 3);
    }

    #[test]
    fn test_streaming_processor() {
        let calculator = LMSRCalculator::new(2, 100.0).unwrap();
        let processor = StreamingProcessor::new(calculator, 2);
        
        // Add first quantities (shouldn't trigger processing)
        let result1 = processor.add_quantities(vec![10.0, 20.0]).unwrap();
        assert!(result1.is_none());
        
        // Add second quantities (should trigger processing)
        let result2 = processor.add_quantities(vec![15.0, 25.0]).unwrap();
        assert!(result2.is_some());
        assert_eq!(result2.unwrap().len(), 2);
    }

    #[test]
    fn test_adaptive_optimizer() {
        let optimizer = AdaptiveOptimizer::new();
        optimizer.record_performance(4, Duration::from_millis(100));
        optimizer.record_performance(8, Duration::from_millis(80));
        
        assert!(optimizer.get_optimal_thread_count() > 0);
        assert!(optimizer.get_optimal_batch_size() > 0);
    }
}