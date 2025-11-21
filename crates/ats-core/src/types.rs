//! Core types and data structures for ATS-CP operations
//!
//! This module provides memory-efficient and high-performance data structures
//! optimized for mathematical operations and real-time processing.

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use std::fmt;

/// High-precision floating-point type for mathematical operations
pub type Precision = f64;

/// Array index type
pub type Index = usize;

/// Time type for performance measurements (nanoseconds)
pub type TimeNs = u64;

/// Temperature value type
pub type Temperature = Precision;

/// Confidence level type (0.0 to 1.0)
pub type Confidence = Precision;

/// Prediction interval as (lower_bound, upper_bound)
pub type PredictionInterval = (Precision, Precision);

/// Vector of prediction intervals
pub type PredictionIntervals = Vec<PredictionInterval>;

/// Calibration scores for conformal prediction
pub type CalibrationScores = Vec<Precision>;

/// Neural network predictions
pub type NeuralPredictions = Vec<Precision>;

/// SIMD-aligned vector for high-performance operations
#[derive(Debug, Clone, PartialEq)]
pub struct AlignedVec<T> {
    data: Vec<T>,
    alignment: usize,
}

impl<T> AlignedVec<T>
where
    T: Clone + Default + Pod + Zeroable,
{
    /// Creates a new aligned vector with specified capacity and alignment
    pub fn new(capacity: usize, alignment: usize) -> Self {
        // Ensure alignment is a power of 2
        assert!(alignment.is_power_of_two(), "Alignment must be a power of 2");
        
        let mut data = Vec::with_capacity(capacity);
        data.resize(capacity, T::default());
        
        Self { data, alignment }
    }

    /// Creates a new aligned vector from existing data
    pub fn from_vec(data: Vec<T>, alignment: usize) -> Self {
        assert!(alignment.is_power_of_two(), "Alignment must be a power of 2");
        Self { data, alignment }
    }

    /// Returns the data slice
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns the mutable data slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Returns the length of the vector
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the alignment requirement
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Resizes the vector to the specified length
    pub fn resize(&mut self, new_len: usize) {
        self.data.resize(new_len, T::default());
    }

    /// Returns the raw pointer to the data
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Returns the raw mutable pointer to the data
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }
}

impl<T> std::ops::Index<usize> for AlignedVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> std::ops::IndexMut<usize> for AlignedVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

/// Performance statistics for ATS-CP operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// Total number of operations performed
    pub total_operations: u64,
    
    /// Total time spent in operations (nanoseconds)
    pub total_time_ns: u64,
    
    /// Average latency per operation (nanoseconds)
    pub average_latency_ns: u64,
    
    /// Minimum latency observed (nanoseconds)
    pub min_latency_ns: u64,
    
    /// Maximum latency observed (nanoseconds)
    pub max_latency_ns: u64,
    
    /// 95th percentile latency (nanoseconds)
    pub p95_latency_ns: u64,
    
    /// 99th percentile latency (nanoseconds)
    pub p99_latency_ns: u64,
    
    /// Operations per second
    pub ops_per_second: f64,
    
    /// Memory usage statistics
    pub memory_usage: MemoryUsage,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    /// Total allocated memory (bytes)
    pub allocated_bytes: u64,
    
    /// Peak memory usage (bytes)
    pub peak_bytes: u64,
    
    /// Current memory usage (bytes)
    pub current_bytes: u64,
    
    /// Number of allocations
    pub allocation_count: u64,
    
    /// Number of deallocations
    pub deallocation_count: u64,
}

/// Temperature scaling result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureScalingResult {
    /// Scaled predictions
    pub scaled_predictions: NeuralPredictions,
    
    /// Optimal temperature found
    pub optimal_temperature: Temperature,
    
    /// Number of iterations used in binary search
    pub iterations: usize,
    
    /// Final tolerance achieved
    pub tolerance: Precision,
    
    /// Execution time (nanoseconds)
    pub execution_time_ns: TimeNs,
}

/// Conformal prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformalPredictionResult {
    /// Prediction intervals
    pub intervals: PredictionIntervals,
    
    /// Confidence level used
    pub confidence: Confidence,
    
    /// Calibration scores used
    pub calibration_scores: CalibrationScores,
    
    /// Quantile threshold
    pub quantile_threshold: Precision,
    
    /// Execution time (nanoseconds)
    pub execution_time_ns: TimeNs,
}

/// Conformal prediction set (placeholder - to be fully implemented)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformalPredictionSet {
    /// Prediction intervals
    pub intervals: PredictionIntervals,
    /// Confidence level
    pub confidence: Confidence,
}

/// SIMD operation result
#[derive(Debug, Clone)]
pub struct SimdResult<T> {
    /// Result data
    pub data: AlignedVec<T>,
    
    /// Number of SIMD operations performed
    pub simd_operations: usize,
    
    /// Execution time (nanoseconds)
    pub execution_time_ns: TimeNs,
}

/// Parallel processing result
#[derive(Debug, Clone)]
pub struct ParallelResult<T> {
    /// Result data
    pub data: Vec<T>,
    
    /// Number of threads used
    pub threads_used: usize,
    
    /// Work distribution efficiency (0.0 to 1.0)
    pub efficiency: f64,
    
    /// Execution time (nanoseconds)
    pub execution_time_ns: TimeNs,
}

/// Integration result with ruv-FANN
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationResult {
    /// Processed predictions
    pub predictions: NeuralPredictions,
    
    /// Neural network metadata
    pub network_metadata: NetworkMetadata,
    
    /// Execution time (nanoseconds)
    pub execution_time_ns: TimeNs,
}

/// Neural network metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetadata {
    /// Number of input neurons
    pub input_neurons: usize,
    
    /// Number of output neurons
    pub output_neurons: usize,
    
    /// Number of hidden layers
    pub hidden_layers: usize,
    
    /// Total number of neurons
    pub total_neurons: usize,
    
    /// Total number of connections
    pub total_connections: usize,
    
    /// Network activation function
    pub activation_function: String,
    
    /// Training algorithm used
    pub training_algorithm: String,
}

/// Latency bucket for histogram tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyBucket {
    /// Upper bound of the bucket (nanoseconds)
    pub upper_bound_ns: u64,
    
    /// Count of operations in this bucket
    pub count: u64,
}

/// Latency histogram for performance monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyHistogram {
    /// Histogram buckets
    pub buckets: Vec<LatencyBucket>,
    
    /// Total count of operations
    pub total_count: u64,
    
    /// Sum of all latencies
    pub sum_ns: u64,
}

impl PerformanceStats {
    /// Creates a new empty performance stats
    pub fn new() -> Self {
        Self {
            total_operations: 0,
            total_time_ns: 0,
            average_latency_ns: 0,
            min_latency_ns: u64::MAX,
            max_latency_ns: 0,
            p95_latency_ns: 0,
            p99_latency_ns: 0,
            ops_per_second: 0.0,
            memory_usage: MemoryUsage::new(),
        }
    }

    /// Updates the statistics with a new operation
    pub fn update(&mut self, latency_ns: u64) {
        self.total_operations += 1;
        self.total_time_ns += latency_ns;
        self.average_latency_ns = self.total_time_ns / self.total_operations;
        self.min_latency_ns = self.min_latency_ns.min(latency_ns);
        self.max_latency_ns = self.max_latency_ns.max(latency_ns);
        
        // Calculate ops per second
        if self.total_time_ns > 0 {
            self.ops_per_second = (self.total_operations as f64) / (self.total_time_ns as f64 / 1_000_000_000.0);
        }
    }

    /// Checks if the current performance meets the target latency
    pub fn meets_latency_target(&self, target_ns: u64) -> bool {
        self.average_latency_ns <= target_ns && self.p99_latency_ns <= target_ns * 2
    }
}

impl MemoryUsage {
    /// Creates a new empty memory usage
    pub fn new() -> Self {
        Self {
            allocated_bytes: 0,
            peak_bytes: 0,
            current_bytes: 0,
            allocation_count: 0,
            deallocation_count: 0,
        }
    }

    /// Records a memory allocation
    pub fn record_allocation(&mut self, bytes: u64) {
        self.allocated_bytes += bytes;
        self.current_bytes += bytes;
        self.peak_bytes = self.peak_bytes.max(self.current_bytes);
        self.allocation_count += 1;
    }

    /// Records a memory deallocation
    pub fn record_deallocation(&mut self, bytes: u64) {
        self.current_bytes = self.current_bytes.saturating_sub(bytes);
        self.deallocation_count += 1;
    }

    /// Returns the current memory efficiency (0.0 to 1.0)
    pub fn efficiency(&self) -> f64 {
        if self.allocation_count == 0 {
            1.0
        } else {
            self.deallocation_count as f64 / self.allocation_count as f64
        }
    }
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for MemoryUsage {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for PerformanceStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PerformanceStats {{ ops: {}, avg_latency: {}ns, p99: {}ns, ops/s: {:.2} }}",
            self.total_operations,
            self.average_latency_ns,
            self.p99_latency_ns,
            self.ops_per_second
        )
    }
}

impl fmt::Display for MemoryUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MemoryUsage {{ allocated: {}MB, peak: {}MB, current: {}MB, efficiency: {:.2}% }}",
            self.allocated_bytes / 1024 / 1024,
            self.peak_bytes / 1024 / 1024,
            self.current_bytes / 1024 / 1024,
            self.efficiency() * 100.0
        )
    }
}

/// ATS-CP algorithm variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AtsCpVariant {
    /// Generalized Quantile: V(x,y) = 1 - softmax(f(x))_y
    GQ,
    /// Adaptive Quantile: V(x,y) = -log(softmax(f(x))_y)
    AQ,
    /// Multi-class Generalized Quantile: V(x,y) = max_{y' ≠ y} softmax(f(x))_{y'}
    MGQ,
    /// Multi-class Adaptive Quantile: Complex multi-class formulation
    MAQ,
}

/// Result of ATS-CP algorithm execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtsCpResult {
    /// Conformal set C_α(x) containing valid predictions
    pub conformal_set: Vec<usize>,
    
    /// Calibrated probabilities p̃(y|x,τ)
    pub calibrated_probabilities: Vec<f64>,
    
    /// Optimal temperature found by SelectTau algorithm
    pub optimal_temperature: Temperature,
    
    /// Quantile threshold q_α used
    pub quantile_threshold: f64,
    
    /// Coverage guarantee (1-α)
    pub coverage_guarantee: Confidence,
    
    /// Execution time in nanoseconds
    pub execution_time_ns: u64,
    
    /// Variant used for computation
    pub variant: AtsCpVariant,
}

/// Mathematical precision validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionValidationResult {
    /// Whether IEEE 754 compliance is maintained
    pub ieee754_compliant: bool,
    
    /// Maximum numerical error detected
    pub max_numerical_error: f64,
    
    /// Whether catastrophic cancellation occurred
    pub catastrophic_cancellation_detected: bool,
    
    /// Condition number for numerical stability
    pub condition_number: f64,
    
    /// Detailed error analysis
    pub error_analysis: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_vec_creation() {
        let vec: AlignedVec<f64> = AlignedVec::new(100, 64);
        assert_eq!(vec.len(), 100);
        assert_eq!(vec.alignment(), 64);
        assert!(!vec.is_empty());
    }

    #[test]
    fn test_aligned_vec_indexing() {
        let mut vec: AlignedVec<f64> = AlignedVec::new(10, 32);
        vec[0] = 1.0;
        vec[9] = 2.0;
        assert_eq!(vec[0], 1.0);
        assert_eq!(vec[9], 2.0);
    }

    #[test]
    fn test_performance_stats_update() {
        let mut stats = PerformanceStats::new();
        stats.update(1000);
        stats.update(2000);
        stats.update(1500);
        
        assert_eq!(stats.total_operations, 3);
        assert_eq!(stats.average_latency_ns, 1500);
        assert_eq!(stats.min_latency_ns, 1000);
        assert_eq!(stats.max_latency_ns, 2000);
        assert!(stats.ops_per_second > 0.0);
    }

    #[test]
    fn test_memory_usage_tracking() {
        let mut usage = MemoryUsage::new();
        usage.record_allocation(1000);
        usage.record_allocation(2000);
        usage.record_deallocation(500);
        
        assert_eq!(usage.allocated_bytes, 3000);
        assert_eq!(usage.current_bytes, 2500);
        assert_eq!(usage.peak_bytes, 3000);
        assert_eq!(usage.allocation_count, 2);
        assert_eq!(usage.deallocation_count, 1);
    }

    #[test]
    fn test_latency_target_check() {
        let mut stats = PerformanceStats::new();
        stats.update(50);
        stats.update(75);
        stats.update(100);
        stats.p99_latency_ns = 150;
        
        assert!(stats.meets_latency_target(100));
        assert!(!stats.meets_latency_target(50));
    }

    #[test]
    fn test_serialization() {
        let stats = PerformanceStats::new();
        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: PerformanceStats = serde_json::from_str(&json).unwrap();
        assert_eq!(stats.total_operations, deserialized.total_operations);
    }
}