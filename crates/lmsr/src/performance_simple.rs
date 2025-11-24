//! Simplified high-performance optimizations for LMSR calculations
//! 
//! This module provides basic optimizations that don't require complex dependencies.

use crate::errors::{LMSRError, Result};
use crate::lmsr::{LMSRCalculator, LMSRMarketMaker};
use crate::market::Market;

use std::sync::Arc;
use std::time::{Duration, Instant};

/// Simple high-frequency trading market maker using basic optimizations
pub struct SimpleHFTMarketMaker {
    calculator: Arc<LMSRCalculator>,
    quantities: Arc<std::sync::RwLock<Vec<f64>>>,
    performance_monitor: Arc<SimplePerformanceMonitor>,
    thread_count: usize,
}

impl SimpleHFTMarketMaker {
    /// Create new simple HFT market maker
    pub fn new(
        num_outcomes: usize,
        liquidity_parameter: f64,
        thread_count: Option<usize>,
    ) -> Result<Self> {
        let calculator = Arc::new(LMSRCalculator::new(num_outcomes, liquidity_parameter)?);
        let quantities = Arc::new(std::sync::RwLock::new(vec![0.0; num_outcomes]));
        let performance_monitor = Arc::new(SimplePerformanceMonitor::new());
        
        Ok(Self {
            calculator,
            quantities,
            performance_monitor,
            thread_count: thread_count.unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)),
        })
    }

    /// Execute trade with performance monitoring
    pub fn execute_trade(&self, trader_id: String, buy_amounts: &[f64]) -> Result<SimpleTradeResult> {
        let start_time = Instant::now();
        
        // Get current prices
        let prices_before = {
            let quantities = self.quantities.read().unwrap();
            self.calculator.all_marginal_prices(&quantities)?
        };
        
        // Calculate trade cost
        let cost = {
            let quantities = self.quantities.read().unwrap();
            self.calculator.calculate_buy_cost(&quantities, buy_amounts)?
        };
        
        // Update quantities
        {
            let mut quantities = self.quantities.write().unwrap();
            for (i, &amount) in buy_amounts.iter().enumerate() {
                quantities[i] += amount;
            }
        }
        
        // Get new prices
        let prices_after = {
            let quantities = self.quantities.read().unwrap();
            self.calculator.all_marginal_prices(&quantities)?
        };
        
        let execution_time = start_time.elapsed();
        self.performance_monitor.record_execution(execution_time);
        
        Ok(SimpleTradeResult {
            trader_id,
            cost,
            prices_before,
            prices_after,
            execution_time,
        })
    }

    /// Get current prices
    pub fn get_prices(&self) -> Result<Vec<f64>> {
        let quantities = self.quantities.read().unwrap();
        self.calculator.all_marginal_prices(&quantities)
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> SimpleHFTStats {
        let monitor_stats = self.performance_monitor.get_stats();
        
        SimpleHFTStats {
            thread_count: self.thread_count,
            average_execution_time: monitor_stats.average_execution_time,
            p99_execution_time: monitor_stats.p99_execution_time,
            trades_per_second: monitor_stats.trades_per_second,
        }
    }
}

/// Simple trade result
#[derive(Debug, Clone)]
pub struct SimpleTradeResult {
    pub trader_id: String,
    pub cost: f64,
    pub prices_before: Vec<f64>,
    pub prices_after: Vec<f64>,
    pub execution_time: Duration,
}

/// Simple performance monitor
pub struct SimplePerformanceMonitor {
    execution_times: std::sync::RwLock<Vec<Duration>>,
    trade_count: std::sync::atomic::AtomicU64,
    start_time: Instant,
}

impl SimplePerformanceMonitor {
    /// Create new simple performance monitor
    pub fn new() -> Self {
        Self {
            execution_times: std::sync::RwLock::new(Vec::new()),
            trade_count: std::sync::atomic::AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    /// Record trade execution time
    pub fn record_execution(&self, duration: Duration) {
        self.trade_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        let mut times = self.execution_times.write().unwrap();
        times.push(duration);
        
        // Keep only recent measurements
        if times.len() > 10000 {
            times.drain(0..5000);
        }
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> SimplePerformanceStats {
        let times = self.execution_times.read().unwrap();
        let trade_count = self.trade_count.load(std::sync::atomic::Ordering::Relaxed);
        let uptime = self.start_time.elapsed();
        
        if times.is_empty() {
            return SimplePerformanceStats {
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

        SimplePerformanceStats {
            average_execution_time: average,
            p99_execution_time: p99,
            trades_per_second,
        }
    }
}

/// Simple performance statistics
#[derive(Debug, Clone)]
pub struct SimplePerformanceStats {
    pub average_execution_time: Duration,
    pub p99_execution_time: Duration,
    pub trades_per_second: f64,
}

/// Simple HFT-specific performance statistics
#[derive(Debug, Clone)]
pub struct SimpleHFTStats {
    pub thread_count: usize,
    pub average_execution_time: Duration,
    pub p99_execution_time: Duration,
    pub trades_per_second: f64,
}

/// Simple batch price calculator
pub struct SimpleBatchCalculator {
    calculator: Arc<LMSRCalculator>,
    thread_count: usize,
}

impl SimpleBatchCalculator {
    /// Create new simple batch calculator
    pub fn new(num_outcomes: usize, liquidity_parameter: f64, thread_count: Option<usize>) -> Result<Self> {
        let calculator = Arc::new(LMSRCalculator::new(num_outcomes, liquidity_parameter)?);
        
        Ok(Self {
            calculator,
            thread_count: thread_count.unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)),
        })
    }

    /// Calculate prices for batch of quantities using simple threading
    pub fn calculate_batch_prices(&self, quantities_batch: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if quantities_batch.is_empty() {
            return Ok(Vec::new());
        }

        if quantities_batch.len() < 100 {
            // Sequential for small batches
            return quantities_batch.iter()
                .map(|quantities| self.calculator.all_marginal_prices(quantities))
                .collect();
        }

        // Parallel for larger batches
        self.calculate_parallel(quantities_batch)
    }

    /// Calculate using standard library threading
    fn calculate_parallel(&self, quantities_batch: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let chunk_size = (quantities_batch.len() / self.thread_count).max(1);
        let mut handles = Vec::new();
        let mut results = vec![Vec::new(); quantities_batch.len()];
        
        for (chunk_idx, chunk) in quantities_batch.chunks(chunk_size).enumerate() {
            let calc = Arc::clone(&self.calculator);
            let chunk_data = chunk.to_vec();
            
            let handle = std::thread::spawn(move || {
                let mut chunk_results = Vec::new();
                for quantities in chunk_data.iter() {
                    match calc.all_marginal_prices(quantities) {
                        Ok(prices) => chunk_results.push(prices),
                        Err(e) => return Err(e),
                    }
                }
                Ok(chunk_results)
            });
            
            handles.push((chunk_idx * chunk_size, handle));
        }
        
        // Collect results
        for (start_idx, handle) in handles {
            let chunk_results = handle.join().map_err(|_| 
                LMSRError::invalid_market("Thread join failed".to_string())
            )??;
            
            for (i, result) in chunk_results.into_iter().enumerate() {
                if start_idx + i < results.len() {
                    results[start_idx + i] = result;
                }
            }
        }
        
        Ok(results)
    }
}

/// Simple streaming processor for continuous price calculations
pub struct SimpleStreamingProcessor {
    calculator: Arc<LMSRCalculator>,
    buffer: std::sync::RwLock<Vec<Vec<f64>>>,
    buffer_size: usize,
    processed_count: std::sync::atomic::AtomicU64,
}

impl SimpleStreamingProcessor {
    /// Create new simple streaming processor
    pub fn new(calculator: LMSRCalculator, buffer_size: usize) -> Self {
        Self {
            calculator: Arc::new(calculator),
            buffer: std::sync::RwLock::new(Vec::with_capacity(buffer_size)),
            buffer_size,
            processed_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Add quantities to processing buffer
    pub fn add_quantities(&self, quantities: Vec<f64>) -> Result<Option<Vec<Vec<f64>>>> {
        let mut buffer = self.buffer.write().unwrap();
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
        let mut buffer = self.buffer.write().unwrap();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_hft_market_maker() {
        let hft_mm = SimpleHFTMarketMaker::new(2, 1000.0, Some(2)).unwrap();
        
        let result = hft_mm.execute_trade("trader1".to_string(), &[10.0, 0.0]).unwrap();
        assert!(result.cost > 0.0);
        assert_eq!(result.trader_id, "trader1");
        assert!(result.execution_time > Duration::new(0, 0));
        
        let stats = hft_mm.get_performance_stats();
        assert_eq!(stats.thread_count, 2);
        assert!(stats.average_execution_time > Duration::new(0, 0));
    }

    #[test]
    fn test_simple_batch_calculator() {
        let calc = SimpleBatchCalculator::new(3, 100.0, Some(2)).unwrap();
        let quantities_batch = vec![
            vec![10.0, 20.0, 30.0],
            vec![5.0, 15.0, 25.0],
        ];
        
        let results = calc.calculate_batch_prices(&quantities_batch).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 3);
        assert_eq!(results[1].len(), 3);
        
        // Verify probabilities sum to 1
        for prices in &results {
            let sum: f64 = prices.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_simple_streaming_processor() {
        let calculator = LMSRCalculator::new(2, 100.0).unwrap();
        let processor = SimpleStreamingProcessor::new(calculator, 2);
        
        // Add first quantities (shouldn't trigger processing)
        let result1 = processor.add_quantities(vec![10.0, 20.0]).unwrap();
        assert!(result1.is_none());
        
        // Add second quantities (should trigger processing)
        let result2 = processor.add_quantities(vec![15.0, 25.0]).unwrap();
        assert!(result2.is_some());
        assert_eq!(result2.unwrap().len(), 2);
        
        assert_eq!(processor.get_processed_count(), 2);
    }

    #[test]
    fn test_simple_performance_monitor() {
        let monitor = SimplePerformanceMonitor::new();
        monitor.record_execution(Duration::from_micros(100));
        monitor.record_execution(Duration::from_micros(150));
        
        let stats = monitor.get_stats();
        assert!(stats.average_execution_time > Duration::new(0, 0));
        assert!(stats.p99_execution_time > Duration::new(0, 0));
    }
}