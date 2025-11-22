//! High-performance optimizations for prospect theory calculations
//! 
//! This module provides memory management, parallel processing, and
//! cache optimization strategies for maximum performance.

use crate::errors::{ProspectTheoryError, Result};
use crate::value_function::{ValueFunction, ValueFunctionParams};
use crate::probability_weighting::{ProbabilityWeighting, WeightingParams};

#[cfg(feature = "simd")]
use crate::simd::{SIMDValueFunction, SIMDProbabilityWeighting, HighPerformanceProspectCalculator};

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::sync::Arc;
use parking_lot::RwLock;
use crossbeam::channel;
use aligned_vec::AVec;
use std::alloc::{GlobalAlloc, Layout};
use std::ptr::NonNull;

/// Custom high-performance allocator for financial calculations
pub struct FinancialAllocator {
    pool: RwLock<Vec<(*mut u8, usize)>>,
    alignment: usize,
}

impl FinancialAllocator {
    /// Create new financial allocator with cache-line alignment
    pub fn new() -> Self {
        Self {
            pool: RwLock::new(Vec::new()),
            alignment: 64, // Cache line alignment
        }
    }

    /// Allocate aligned memory for SIMD operations
    pub fn allocate_aligned(&self, size: usize, alignment: usize) -> Option<NonNull<u8>> {
        let layout = Layout::from_size_align(size, alignment).ok()?;
        
        // Try to reuse from pool first
        {
            let mut pool = self.pool.write();
            if let Some(index) = pool.iter().position(|(_, pool_size)| *pool_size >= size) {
                let (ptr, _) = pool.remove(index);
                return NonNull::new(ptr);
            }
        }
        
        // Allocate new if pool empty
        unsafe {
            let ptr = std::alloc::alloc(layout);
            NonNull::new(ptr)
        }
    }

    /// Return memory to pool for reuse
    pub fn deallocate_to_pool(&self, ptr: NonNull<u8>, size: usize) {
        let mut pool = self.pool.write();
        if pool.len() < 64 { // Limit pool size
            pool.push((ptr.as_ptr(), size));
        } else {
            unsafe {
                let layout = Layout::from_size_align_unchecked(size, self.alignment);
                std::alloc::dealloc(ptr.as_ptr(), layout);
            }
        }
    }
}

/// Lock-free object pool for frequently used calculations
pub struct ObjectPool<T> {
    objects: crossbeam::queue::SegQueue<T>,
    factory: Box<dyn Fn() -> T + Send + Sync>,
    max_size: usize,
}

impl<T> ObjectPool<T> {
    /// Create new object pool
    pub fn new<F>(factory: F, max_size: usize) -> Self 
    where 
        F: Fn() -> T + Send + Sync + 'static
    {
        let pool = Self {
            objects: crossbeam::queue::SegQueue::new(),
            factory: Box::new(factory),
            max_size,
        };
        
        // Pre-populate pool
        for _ in 0..max_size / 2 {
            pool.objects.push((pool.factory)());
        }
        
        pool
    }

    /// Get object from pool or create new
    pub fn get(&self) -> T {
        self.objects.pop().unwrap_or_else(|| (self.factory)())
    }

    /// Return object to pool
    pub fn put(&self, obj: T) {
        if self.objects.len() < self.max_size {
            self.objects.push(obj);
        }
        // Drop if pool is full
    }
}

/// High-performance batch processor for prospect theory calculations
pub struct BatchProcessor {
    value_function: Arc<ValueFunction>,
    probability_weighting: Arc<ProbabilityWeighting>,
    #[cfg(feature = "simd")]
    simd_calculator: Option<Arc<HighPerformanceProspectCalculator>>,
    thread_pool: rayon::ThreadPool,
    memory_allocator: Arc<FinancialAllocator>,
    result_pool: Arc<ObjectPool<AVec<f64>>>,
}

impl BatchProcessor {
    /// Create new batch processor with optimized thread pool
    pub fn new(
        value_params: ValueFunctionParams,
        weighting_params: WeightingParams,
        num_threads: Option<usize>,
    ) -> Result<Self> {
        let value_function = Arc::new(ValueFunction::new(value_params.clone())?);
        let probability_weighting = Arc::new(ProbabilityWeighting::new(weighting_params.clone(), 
            crate::probability_weighting::WeightingFunction::TverskyKahneman)?);

        // Create optimized thread pool
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads.unwrap_or_else(|| num_cpus::get()))
            .stack_size(8 * 1024 * 1024) // 8MB stack for complex calculations
            .build()
            .map_err(|e| ProspectTheoryError::computation_failed(&format!("Thread pool creation failed: {}", e)))?;

        let memory_allocator = Arc::new(FinancialAllocator::new());
        let result_pool = Arc::new(ObjectPool::new(
            || AVec::with_capacity(1000),
            32
        ));

        #[cfg(feature = "simd")]
        let simd_calculator = Some(Arc::new(
            HighPerformanceProspectCalculator::new(value_params, weighting_params)?
        ));
        #[cfg(not(feature = "simd"))]
        let simd_calculator = None;

        Ok(Self {
            value_function,
            probability_weighting,
            simd_calculator,
            thread_pool,
            memory_allocator,
            result_pool,
        })
    }

    /// Process large batch of prospect calculations with maximum performance
    pub fn process_batch(
        &self,
        outcomes_batch: &[Vec<f64>],
        probabilities_batch: &[Vec<f64>],
    ) -> Result<Vec<f64>> {
        if outcomes_batch.len() != probabilities_batch.len() {
            return Err(ProspectTheoryError::computation_failed(
                "Outcomes and probabilities batch lengths must match"
            ));
        }

        // Choose optimal processing strategy based on data size
        let total_calculations: usize = outcomes_batch.iter().map(|v| v.len()).sum();
        
        if total_calculations > 100_000 {
            self.process_large_batch(outcomes_batch, probabilities_batch)
        } else if total_calculations > 10_000 {
            self.process_medium_batch(outcomes_batch, probabilities_batch)
        } else {
            self.process_small_batch(outcomes_batch, probabilities_batch)
        }
    }

    /// Process very large batches with memory optimization
    fn process_large_batch(
        &self,
        outcomes_batch: &[Vec<f64>],
        probabilities_batch: &[Vec<f64>],
    ) -> Result<Vec<f64>> {
        // Use streaming processing to avoid memory pressure
        let chunk_size = 1000;
        let mut results = Vec::with_capacity(outcomes_batch.len());

        for chunk_outcomes in outcomes_batch.chunks(chunk_size) {
            let chunk_probs = &probabilities_batch[results.len()..results.len() + chunk_outcomes.len()];
            
            let chunk_results = self.thread_pool.install(|| {
                chunk_outcomes
                    .par_iter()
                    .zip(chunk_probs.par_iter())
                    .map(|(outcomes, probabilities)| {
                        self.calculate_single_prospect(outcomes, probabilities)
                    })
                    .collect::<Result<Vec<f64>>>()
            })?;
            
            results.extend(chunk_results);
        }

        Ok(results)
    }

    /// Process medium batches with parallel optimization
    fn process_medium_batch(
        &self,
        outcomes_batch: &[Vec<f64>],
        probabilities_batch: &[Vec<f64>],
    ) -> Result<Vec<f64>> {
        self.thread_pool.install(|| {
            outcomes_batch
                .par_iter()
                .zip(probabilities_batch.par_iter())
                .map(|(outcomes, probabilities)| {
                    self.calculate_single_prospect(outcomes, probabilities)
                })
                .collect()
        })
    }

    /// Process small batches with minimal overhead
    fn process_small_batch(
        &self,
        outcomes_batch: &[Vec<f64>],
        probabilities_batch: &[Vec<f64>],
    ) -> Result<Vec<f64>> {
        outcomes_batch
            .iter()
            .zip(probabilities_batch.iter())
            .map(|(outcomes, probabilities)| {
                self.calculate_single_prospect(outcomes, probabilities)
            })
            .collect()
    }

    /// Calculate single prospect value with optimal method selection
    fn calculate_single_prospect(&self, outcomes: &[f64], probabilities: &[f64]) -> Result<f64> {
        #[cfg(feature = "simd")]
        if let Some(ref simd_calc) = self.simd_calculator {
            if outcomes.len() >= 8 { // SIMD worthwhile for larger calculations
                return simd_calc.calculate_prospect_value(outcomes, probabilities);
            }
        }

        // Fallback to standard calculation
        let values = self.value_function.values(outcomes)?;
        let decision_weights = self.probability_weighting.decision_weights(probabilities, outcomes)?;
        
        let prospect_value: f64 = values
            .iter()
            .zip(decision_weights.iter())
            .map(|(&value, &weight)| value * weight)
            .sum();

        Ok(prospect_value)
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> PerformanceStats {
        PerformanceStats {
            thread_count: self.thread_pool.current_num_threads(),
            memory_pool_size: self.result_pool.objects.len(),
            simd_enabled: self.simd_calculator.is_some(),
        }
    }
}

/// Performance monitoring and statistics
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub thread_count: usize,
    pub memory_pool_size: usize,
    pub simd_enabled: bool,
}

/// Cache-optimized data structures for hot path calculations
pub struct CacheOptimizedCalculator {
    // Pre-computed lookup tables for common values
    power_cache: RwLock<std::collections::HashMap<(u64, u64), f64>>,
    log_cache: RwLock<std::collections::HashMap<u64, f64>>,
    exp_cache: RwLock<std::collections::HashMap<u64, f64>>,
    cache_hits: std::sync::atomic::AtomicU64,
    cache_misses: std::sync::atomic::AtomicU64,
}

impl CacheOptimizedCalculator {
    /// Create new cache-optimized calculator
    pub fn new() -> Self {
        Self {
            power_cache: RwLock::new(std::collections::HashMap::with_capacity(10000)),
            log_cache: RwLock::new(std::collections::HashMap::with_capacity(1000)),
            exp_cache: RwLock::new(std::collections::HashMap::with_capacity(1000)),
            cache_hits: std::sync::atomic::AtomicU64::new(0),
            cache_misses: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Fast cached power calculation
    pub fn cached_pow(&self, base: f64, exponent: f64) -> f64 {
        let base_bits = base.to_bits();
        let exp_bits = exponent.to_bits();
        let key = (base_bits, exp_bits);

        // Try cache first
        {
            let cache = self.power_cache.read();
            if let Some(&result) = cache.get(&key) {
                self.cache_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return result;
            }
        }

        // Compute and cache
        let result = base.powf(exponent);
        {
            let mut cache = self.power_cache.write();
            if cache.len() < 10000 { // Limit cache size
                cache.insert(key, result);
            }
        }
        
        self.cache_misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        result
    }

    /// Get cache hit ratio
    pub fn cache_hit_ratio(&self) -> f64 {
        let hits = self.cache_hits.load(std::sync::atomic::Ordering::Relaxed);
        let misses = self.cache_misses.load(std::sync::atomic::Ordering::Relaxed);
        
        if hits + misses == 0 {
            0.0
        } else {
            hits as f64 / (hits + misses) as f64
        }
    }
}

/// Memory bandwidth optimizer for large-scale calculations
pub struct MemoryBandwidthOptimizer {
    prefetch_distance: usize,
    chunk_size: usize,
}

impl MemoryBandwidthOptimizer {
    /// Create new memory bandwidth optimizer
    pub fn new() -> Self {
        Self {
            prefetch_distance: 64, // Cache lines ahead
            chunk_size: 4096,     // Process in 4KB chunks
        }
    }

    /// Optimize memory access pattern for vectorized calculations
    pub fn optimize_access_pattern<T>(&self, data: &[T]) -> Vec<usize> {
        // Generate optimal access indices for cache efficiency
        let mut indices = Vec::with_capacity(data.len());
        
        // Process in cache-friendly chunks
        for chunk_start in (0..data.len()).step_by(self.chunk_size) {
            let chunk_end = (chunk_start + self.chunk_size).min(data.len());
            for i in chunk_start..chunk_end {
                indices.push(i);
            }
        }
        
        indices
    }

    /// Prefetch memory for upcoming calculations
    #[cfg(target_arch = "x86_64")]
    pub fn prefetch_memory(&self, ptr: *const u8) {
        unsafe {
            use std::arch::x86_64::_mm_prefetch;
            _mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn prefetch_memory(&self, _ptr: *const u8) {
        // No-op on non-x86_64 platforms
    }
}

/// Adaptive performance tuner that optimizes parameters based on workload
pub struct AdaptivePerformanceTuner {
    execution_times: RwLock<Vec<std::time::Duration>>,
    optimal_thread_count: std::sync::atomic::AtomicUsize,
    optimal_batch_size: std::sync::atomic::AtomicUsize,
}

impl AdaptivePerformanceTuner {
    /// Create new adaptive performance tuner
    pub fn new() -> Self {
        Self {
            execution_times: RwLock::new(Vec::new()),
            optimal_thread_count: std::sync::atomic::AtomicUsize::new(num_cpus::get()),
            optimal_batch_size: std::sync::atomic::AtomicUsize::new(1000),
        }
    }

    /// Record execution time for performance tuning
    pub fn record_execution(&self, duration: std::time::Duration) {
        let mut times = self.execution_times.write();
        times.push(duration);
        
        // Keep only recent measurements
        if times.len() > 100 {
            times.drain(0..50);
        }
    }

    /// Get optimal thread count based on recent performance
    pub fn get_optimal_thread_count(&self) -> usize {
        self.optimal_thread_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get optimal batch size based on recent performance
    pub fn get_optimal_batch_size(&self) -> usize {
        self.optimal_batch_size.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Tune parameters based on performance history
    pub fn tune_parameters(&self) {
        let times = self.execution_times.read();
        if times.len() < 10 {
            return;
        }

        // Simple heuristic: if average time is high, try different parameters
        let avg_time: std::time::Duration = times.iter().sum::<std::time::Duration>() / times.len() as u32;
        
        if avg_time > std::time::Duration::from_millis(100) {
            // Try more threads for CPU-bound work
            let current_threads = self.optimal_thread_count.load(std::sync::atomic::Ordering::Relaxed);
            let new_threads = (current_threads * 2).min(num_cpus::get() * 2);
            self.optimal_thread_count.store(new_threads, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value_function::ValueFunctionParams;
    use crate::probability_weighting::WeightingParams;

    #[test]
    fn test_financial_allocator() {
        let allocator = FinancialAllocator::new();
        let ptr = allocator.allocate_aligned(1024, 64).unwrap();
        allocator.deallocate_to_pool(ptr, 1024);
    }

    #[test]
    fn test_object_pool() {
        let pool = ObjectPool::new(|| Vec::<f64>::new(), 10);
        let obj1 = pool.get();
        let obj2 = pool.get();
        pool.put(obj1);
        pool.put(obj2);
    }

    #[test]
    fn test_batch_processor() {
        let value_params = ValueFunctionParams::default();
        let weighting_params = WeightingParams::default();
        let processor = BatchProcessor::new(value_params, weighting_params, Some(2)).unwrap();
        
        let outcomes_batch = vec![
            vec![100.0, 0.0, -100.0],
            vec![50.0, -50.0],
        ];
        let probabilities_batch = vec![
            vec![0.3, 0.4, 0.3],
            vec![0.5, 0.5],
        ];
        
        let results = processor.process_batch(&outcomes_batch, &probabilities_batch).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_cache_optimized_calculator() {
        let calc = CacheOptimizedCalculator::new();
        
        let result1 = calc.cached_pow(2.0, 3.0);
        let result2 = calc.cached_pow(2.0, 3.0); // Should hit cache
        
        assert_eq!(result1, result2);
        assert!(calc.cache_hit_ratio() > 0.0);
    }

    #[test]
    fn test_memory_bandwidth_optimizer() {
        let optimizer = MemoryBandwidthOptimizer::new();
        let data = vec![1, 2, 3, 4, 5];
        let indices = optimizer.optimize_access_pattern(&data);
        assert_eq!(indices.len(), data.len());
    }

    #[test]
    fn test_adaptive_performance_tuner() {
        let tuner = AdaptivePerformanceTuner::new();
        tuner.record_execution(std::time::Duration::from_millis(50));
        tuner.record_execution(std::time::Duration::from_millis(75));
        
        assert!(tuner.get_optimal_thread_count() > 0);
        assert!(tuner.get_optimal_batch_size() > 0);
    }
}