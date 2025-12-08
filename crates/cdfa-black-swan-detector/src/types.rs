//! Core types and data structures for Black Swan detection

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Instant;

/// Price type for high-precision calculations
pub type Price = f64;

/// Volume type for market data
pub type Volume = f64;

/// Probability type for risk calculations
pub type Probability = f64;

/// Timestamp type for real-time processing
pub type Timestamp = u64;

/// Market data point for Black Swan analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub timestamp: Timestamp,
    pub open: Price,
    pub high: Price,
    pub low: Price,
    pub close: Price,
    pub volume: Volume,
}

/// Return data for statistical analysis
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ReturnData {
    pub log_return: f64,
    pub abs_return: f64,
    pub squared_return: f64,
    pub timestamp: Timestamp,
}

/// Extreme Value Theory parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EVTParameters {
    /// Hill estimator parameter k (number of order statistics)
    pub k: usize,
    /// Tail threshold quantile (e.g., 0.95 for 95th percentile)
    pub threshold: f64,
    /// Minimum number of observations for reliable estimation
    pub min_observations: usize,
    /// Confidence level for statistical tests
    pub confidence_level: f64,
}

impl Default for EVTParameters {
    fn default() -> Self {
        Self {
            k: 100,
            threshold: 0.95,
            min_observations: 50,
            confidence_level: 0.05,
        }
    }
}

/// Tail risk metrics from EVT analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TailRiskMetrics {
    /// Hill estimator value (tail index)
    pub hill_estimator: f64,
    /// Value at Risk (VaR) at specified confidence level
    pub var: f64,
    /// Expected Shortfall (Conditional VaR)
    pub expected_shortfall: f64,
    /// Tail probability estimate
    pub tail_probability: f64,
    /// Statistical significance p-value
    pub p_value: f64,
    /// Number of observations used
    pub n_observations: usize,
}

/// Black Swan detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackSwanResult {
    /// Overall Black Swan probability [0, 1]
    pub probability: Probability,
    /// Confidence in the prediction [0, 1]
    pub confidence: f64,
    /// Expected direction: -1 (down), 0 (neutral), 1 (up)
    pub direction: i8,
    /// Severity estimate: 0 (low) to 1 (extreme)
    pub severity: f64,
    /// Time horizon in milliseconds
    pub time_horizon: u64,
    /// Component probabilities
    pub components: BlackSwanComponents,
    /// Computational metrics
    pub metrics: ComputationMetrics,
}

/// Component probabilities contributing to Black Swan detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackSwanComponents {
    /// Fat tail probability from EVT
    pub fat_tail: f64,
    /// Volatility clustering score
    pub volatility_clustering: f64,
    /// Liquidity crisis probability
    pub liquidity_crisis: f64,
    /// Correlation breakdown probability
    pub correlation_breakdown: f64,
    /// Jump discontinuity probability
    pub jump_discontinuity: f64,
    /// Market microstructure anomaly score
    pub microstructure_anomaly: f64,
}

/// Computational performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationMetrics {
    /// Total computation time in nanoseconds
    pub computation_time_ns: u64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Number of SIMD operations performed
    pub simd_operations: usize,
    /// GPU utilization percentage
    pub gpu_utilization: f32,
    /// Cache hit ratio
    pub cache_hit_ratio: f32,
}

/// Rolling window buffer for efficient data management
#[derive(Debug, Clone)]
pub struct RollingWindow<T> {
    data: VecDeque<T>,
    capacity: usize,
    sum: f64,
    sum_squares: f64,
}

impl<T> RollingWindow<T>
where
    T: Clone + Into<f64>,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(capacity),
            capacity,
            sum: 0.0,
            sum_squares: 0.0,
        }
    }

    pub fn push(&mut self, value: T) {
        let val_f64 = value.clone().into();
        
        if self.data.len() == self.capacity {
            if let Some(old_val) = self.data.pop_front() {
                let old_f64 = old_val.into();
                self.sum -= old_f64;
                self.sum_squares -= old_f64 * old_f64;
            }
        }
        
        self.data.push_back(value);
        self.sum += val_f64;
        self.sum_squares += val_f64 * val_f64;
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn mean(&self) -> f64 {
        if self.data.is_empty() {
            0.0
        } else {
            self.sum / self.data.len() as f64
        }
    }

    pub fn variance(&self) -> f64 {
        if self.data.len() < 2 {
            0.0
        } else {
            let n = self.data.len() as f64;
            let mean = self.mean();
            (self.sum_squares / n) - (mean * mean)
        }
    }

    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    pub fn as_slice(&self) -> &[T] {
        self.data.as_slices().0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn is_full(&self) -> bool {
        self.data.len() == self.capacity
    }
}

impl Into<f64> for MarketData {
    fn into(self) -> f64 {
        self.close
    }
}

impl Into<f64> for ReturnData {
    fn into(self) -> f64 {
        self.log_return
    }
}

/// Memory-efficient circular buffer for high-frequency data
#[derive(Debug)]
pub struct CircularBuffer<T> {
    buffer: Vec<T>,
    head: usize,
    tail: usize,
    size: usize,
    capacity: usize,
}

impl<T: Clone + Default> CircularBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![T::default(); capacity],
            head: 0,
            tail: 0,
            size: 0,
            capacity,
        }
    }

    pub fn push(&mut self, item: T) {
        self.buffer[self.tail] = item;
        self.tail = (self.tail + 1) % self.capacity;
        
        if self.size < self.capacity {
            self.size += 1;
        } else {
            self.head = (self.head + 1) % self.capacity;
        }
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn is_full(&self) -> bool {
        self.size == self.capacity
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.size {
            let actual_index = (self.head + index) % self.capacity;
            Some(&self.buffer[actual_index])
        } else {
            None
        }
    }

    pub fn iter(&self) -> CircularBufferIterator<T> {
        CircularBufferIterator {
            buffer: self,
            index: 0,
        }
    }
}

/// Iterator for CircularBuffer
pub struct CircularBufferIterator<'a, T> {
    buffer: &'a CircularBuffer<T>,
    index: usize,
}

impl<'a, T> Iterator for CircularBufferIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.buffer.len() {
            let item = self.buffer.get(self.index);
            self.index += 1;
            item
        } else {
            None
        }
    }
}

/// Performance timer for benchmarking
#[derive(Debug)]
pub struct PerformanceTimer {
    start: Instant,
    name: String,
}

impl PerformanceTimer {
    pub fn new(name: &str) -> Self {
        Self {
            start: Instant::now(),
            name: name.to_string(),
        }
    }

    pub fn elapsed_nanos(&self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }

    pub fn elapsed_micros(&self) -> u64 {
        self.start.elapsed().as_micros() as u64
    }

    pub fn elapsed_millis(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }
}

impl Drop for PerformanceTimer {
    fn drop(&mut self) {
        let elapsed = self.elapsed_nanos();
        if elapsed < 1_000 {
            log::debug!("{}: {}ns", self.name, elapsed);
        } else if elapsed < 1_000_000 {
            log::debug!("{}: {}μs", self.name, elapsed / 1_000);
        } else {
            log::debug!("{}: {}ms", self.name, elapsed / 1_000_000);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_rolling_window() {
        let mut window = RollingWindow::new(3);
        assert_eq!(window.len(), 0);
        assert!(window.is_empty());

        window.push(1.0);
        window.push(2.0);
        window.push(3.0);
        
        assert_eq!(window.len(), 3);
        assert_relative_eq!(window.mean(), 2.0);
        assert_relative_eq!(window.variance(), 2.0/3.0);

        window.push(4.0); // Should remove 1.0
        assert_eq!(window.len(), 3);
        assert_relative_eq!(window.mean(), 3.0);
    }

    #[test]
    fn test_circular_buffer() {
        let mut buffer = CircularBuffer::new(3);
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());

        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        
        assert_eq!(buffer.len(), 3);
        assert!(buffer.is_full());
        assert_eq!(buffer.get(0), Some(&1));
        assert_eq!(buffer.get(1), Some(&2));
        assert_eq!(buffer.get(2), Some(&3));

        buffer.push(4); // Should overwrite 1
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.get(0), Some(&2));
        assert_eq!(buffer.get(1), Some(&3));
        assert_eq!(buffer.get(2), Some(&4));
    }

    #[test]
    fn test_performance_timer() {
        let timer = PerformanceTimer::new("test");
        std::thread::sleep(std::time::Duration::from_millis(1));
        assert!(timer.elapsed_nanos() > 0);
        assert!(timer.elapsed_micros() > 0);
    }
}