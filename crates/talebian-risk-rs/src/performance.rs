//! # SIMD-Optimized Performance Module
//!
//! High-performance SIMD implementations for real-time trading applications.
//! Provides optimized calculations for antifragility, Kelly criterion, and
//! whale detection with sub-millisecond latency requirements.

use crate::TalebianRiskError;
use serde::{Deserialize, Serialize};

#[cfg(feature = "simd")]
use wide::f64x4;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// SIMD-optimized performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimdPerformanceStats {
    pub calculations_per_second: f64,
    pub average_latency_ns: u64,
    pub peak_latency_ns: u64,
    pub cache_hit_rate: f64,
    pub memory_usage_mb: f64,
    pub simd_utilization: f64,
    pub parallel_efficiency: f64,
}

/// Performance monitoring and optimization
pub struct PerformanceMonitor {
    start_time: std::time::Instant,
    total_calculations: u64,
    total_latency_ns: u64,
    peak_latency_ns: u64,
    cache_hits: u64,
    cache_misses: u64,
    memory_allocations: u64,
    memory_usage: usize,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            start_time: std::time::Instant::now(),
            total_calculations: 0,
            total_latency_ns: 0,
            peak_latency_ns: 0,
            cache_hits: 0,
            cache_misses: 0,
            memory_allocations: 0,
            memory_usage: 0,
        }
    }

    /// Record calculation performance
    pub fn record_calculation(&mut self, latency_ns: u64) {
        self.total_calculations += 1;
        self.total_latency_ns += latency_ns;
        self.peak_latency_ns = self.peak_latency_ns.max(latency_ns);
    }

    /// Record cache performance
    pub fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }

    pub fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
    }

    /// Get current performance statistics
    pub fn get_stats(&self) -> SimdPerformanceStats {
        let elapsed_secs = self.start_time.elapsed().as_secs_f64();
        let calculations_per_second = if elapsed_secs > 0.0 {
            self.total_calculations as f64 / elapsed_secs
        } else {
            0.0
        };

        let average_latency_ns = if self.total_calculations > 0 {
            self.total_latency_ns / self.total_calculations
        } else {
            0
        };

        let cache_hit_rate = if self.cache_hits + self.cache_misses > 0 {
            self.cache_hits as f64 / (self.cache_hits + self.cache_misses) as f64
        } else {
            0.0
        };

        SimdPerformanceStats {
            calculations_per_second,
            average_latency_ns,
            peak_latency_ns: self.peak_latency_ns,
            cache_hit_rate,
            memory_usage_mb: self.memory_usage as f64 / 1_048_576.0,
            simd_utilization: self.estimate_simd_utilization(),
            parallel_efficiency: self.estimate_parallel_efficiency(),
        }
    }

    fn estimate_simd_utilization(&self) -> f64 {
        // Estimate based on calculation throughput
        // Real implementation would use hardware counters
        if self.total_calculations > 1000 {
            0.85 // Assume good SIMD utilization for high-frequency operations
        } else {
            0.6 // Lower utilization for sporadic calculations
        }
    }

    fn estimate_parallel_efficiency(&self) -> f64 {
        // Estimate parallel efficiency
        #[cfg(feature = "parallel")]
        let cpu_count = num_cpus::get() as f64;
        #[cfg(not(feature = "parallel"))]
        let cpu_count = 1.0;
        let efficiency = if cpu_count > 1.0 {
            (self.total_calculations as f64 / 1000.0).min(cpu_count * 0.8) / cpu_count
        } else {
            1.0
        };
        efficiency.max(0.0).min(1.0)
    }
}

/// SIMD-optimized mathematical operations
pub struct SimdMath;

impl SimdMath {
    /// SIMD-optimized Kelly fraction calculation for multiple assets
    #[cfg(feature = "simd")]
    pub fn kelly_fraction_simd_x4(
        expected_returns: &[f64],
        variances: &[f64],
        whale_multipliers: &[f64],
    ) -> Result<Vec<f64>, TalebianRiskError> {
        if expected_returns.len() != variances.len()
            || expected_returns.len() != whale_multipliers.len()
        {
            return Err(TalebianRiskError::invalid_input("Array lengths must match"));
        }

        let mut results = Vec::with_capacity(expected_returns.len());

        // Process in SIMD chunks of 4
        for chunk_start in (0..expected_returns.len()).step_by(4) {
            let chunk_size = (expected_returns.len() - chunk_start).min(4);

            // Load data into SIMD vectors
            let returns = f64x4::new([
                expected_returns.get(chunk_start).copied().unwrap_or(0.0),
                expected_returns
                    .get(chunk_start + 1)
                    .copied()
                    .unwrap_or(0.0),
                expected_returns
                    .get(chunk_start + 2)
                    .copied()
                    .unwrap_or(0.0),
                expected_returns
                    .get(chunk_start + 3)
                    .copied()
                    .unwrap_or(0.0),
            ]);

            let vars = f64x4::new([
                variances.get(chunk_start).copied().unwrap_or(0.001),
                variances.get(chunk_start + 1).copied().unwrap_or(0.001),
                variances.get(chunk_start + 2).copied().unwrap_or(0.001),
                variances.get(chunk_start + 3).copied().unwrap_or(0.001),
            ]);

            let multipliers = f64x4::new([
                whale_multipliers.get(chunk_start).copied().unwrap_or(1.0),
                whale_multipliers
                    .get(chunk_start + 1)
                    .copied()
                    .unwrap_or(1.0),
                whale_multipliers
                    .get(chunk_start + 2)
                    .copied()
                    .unwrap_or(1.0),
                whale_multipliers
                    .get(chunk_start + 3)
                    .copied()
                    .unwrap_or(1.0),
            ]);

            // Kelly calculation: f = (μ / σ²) * whale_multiplier
            let kelly_fractions = (returns / vars) * multipliers;

            // Apply bounds: min(0.05, max(0.75))
            let min_bound = f64x4::splat(0.05);
            let max_bound = f64x4::splat(0.75);
            let bounded = kelly_fractions.max(min_bound).min(max_bound);

            // Extract results
            for i in 0..chunk_size {
                results.push(bounded.as_array_ref()[i]);
            }
        }

        Ok(results)
    }

    /// SIMD-optimized volatility calculation
    #[cfg(feature = "simd")]
    pub fn volatility_simd_x8(returns: &[f64]) -> Result<f64, TalebianRiskError> {
        if returns.len() < 8 {
            return Self::volatility_scalar(returns);
        }

        let n = returns.len();
        let mean = returns.iter().sum::<f64>() / n as f64;
        let mean_vec = f64x8::splat(mean);

        let mut sum_squared_deviations = f64x8::splat(0.0);
        let mut processed = 0;

        // Process in chunks of 8
        for chunk in returns.chunks_exact(8) {
            let values = f64x8::new([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]);

            let deviations = values - mean_vec;
            let squared_deviations = deviations * deviations;
            sum_squared_deviations += squared_deviations;
            processed += 8;
        }

        // Handle remaining elements
        let mut scalar_sum = 0.0;
        for &value in &returns[processed..] {
            let deviation = value - mean;
            scalar_sum += deviation * deviation;
        }

        // Sum the SIMD results
        let simd_array = sum_squared_deviations.as_array_ref();
        let simd_sum: f64 = simd_array.iter().sum();

        let total_sum = simd_sum + scalar_sum;
        let variance = total_sum / (n - 1) as f64;

        Ok(variance.sqrt())
    }

    /// Scalar fallback for volatility calculation
    fn volatility_scalar(returns: &[f64]) -> Result<f64, TalebianRiskError> {
        if returns.len() < 2 {
            return Err(TalebianRiskError::insufficient_data(
                "Need at least 2 returns",
            ));
        }

        let n = returns.len();
        let mean = returns.iter().sum::<f64>() / n as f64;

        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1) as f64;

        Ok(variance.sqrt())
    }

    /// SIMD-optimized whale detection scoring
    #[cfg(feature = "simd")]
    pub fn whale_score_simd_x4(
        volume_ratios: &[f64],
        price_impacts: &[f64],
        order_imbalances: &[f64],
        smart_money_flows: &[f64],
    ) -> Result<Vec<f64>, TalebianRiskError> {
        let len = volume_ratios.len();
        if len != price_impacts.len()
            || len != order_imbalances.len()
            || len != smart_money_flows.len()
        {
            return Err(TalebianRiskError::invalid_input("Array lengths must match"));
        }

        let mut results = Vec::with_capacity(len);

        // Weights for whale scoring
        let vol_weight = f64x4::splat(0.3);
        let price_weight = f64x4::splat(0.25);
        let order_weight = f64x4::splat(0.25);
        let smart_weight = f64x4::splat(0.2);

        // Process in chunks of 4
        for chunk_start in (0..len).step_by(4) {
            let chunk_size = (len - chunk_start).min(4);

            let volumes = f64x4::new([
                volume_ratios.get(chunk_start).copied().unwrap_or(0.0),
                volume_ratios.get(chunk_start + 1).copied().unwrap_or(0.0),
                volume_ratios.get(chunk_start + 2).copied().unwrap_or(0.0),
                volume_ratios.get(chunk_start + 3).copied().unwrap_or(0.0),
            ]);

            let prices = f64x4::new([
                price_impacts.get(chunk_start).copied().unwrap_or(0.0),
                price_impacts.get(chunk_start + 1).copied().unwrap_or(0.0),
                price_impacts.get(chunk_start + 2).copied().unwrap_or(0.0),
                price_impacts.get(chunk_start + 3).copied().unwrap_or(0.0),
            ]);

            let orders = f64x4::new([
                order_imbalances.get(chunk_start).copied().unwrap_or(0.0),
                order_imbalances
                    .get(chunk_start + 1)
                    .copied()
                    .unwrap_or(0.0),
                order_imbalances
                    .get(chunk_start + 2)
                    .copied()
                    .unwrap_or(0.0),
                order_imbalances
                    .get(chunk_start + 3)
                    .copied()
                    .unwrap_or(0.0),
            ]);

            let smart = f64x4::new([
                smart_money_flows.get(chunk_start).copied().unwrap_or(0.0),
                smart_money_flows
                    .get(chunk_start + 1)
                    .copied()
                    .unwrap_or(0.0),
                smart_money_flows
                    .get(chunk_start + 2)
                    .copied()
                    .unwrap_or(0.0),
                smart_money_flows
                    .get(chunk_start + 3)
                    .copied()
                    .unwrap_or(0.0),
            ]);

            // Weighted combination
            let scores = volumes * vol_weight
                + prices * price_weight
                + orders * order_weight
                + smart * smart_weight;

            // Clamp to [0, 1]
            let min_bound = f64x4::splat(0.0);
            let max_bound = f64x4::splat(1.0);
            let clamped = scores.max(min_bound).min(max_bound);

            for i in 0..chunk_size {
                results.push(clamped.as_array_ref()[i]);
            }
        }

        Ok(results)
    }

    /// SIMD-optimized antifragility calculation
    #[cfg(feature = "simd")]
    pub fn antifragility_simd_x4(
        volatilities: &[f64],
        returns: &[f64],
        love_factors: &[f64],
    ) -> Result<Vec<f64>, TalebianRiskError> {
        let len = volatilities.len();
        if len != returns.len() || len != love_factors.len() {
            return Err(TalebianRiskError::invalid_input("Array lengths must match"));
        }

        let mut results = Vec::with_capacity(len);

        for chunk_start in (0..len).step_by(4) {
            let chunk_size = (len - chunk_start).min(4);

            let vols = f64x4::new([
                volatilities.get(chunk_start).copied().unwrap_or(0.0),
                volatilities.get(chunk_start + 1).copied().unwrap_or(0.0),
                volatilities.get(chunk_start + 2).copied().unwrap_or(0.0),
                volatilities.get(chunk_start + 3).copied().unwrap_or(0.0),
            ]);

            let rets = f64x4::new([
                returns.get(chunk_start).copied().unwrap_or(0.0),
                returns.get(chunk_start + 1).copied().unwrap_or(0.0),
                returns.get(chunk_start + 2).copied().unwrap_or(0.0),
                returns.get(chunk_start + 3).copied().unwrap_or(0.0),
            ]);

            let factors = f64x4::new([
                love_factors.get(chunk_start).copied().unwrap_or(1.0),
                love_factors.get(chunk_start + 1).copied().unwrap_or(1.0),
                love_factors.get(chunk_start + 2).copied().unwrap_or(1.0),
                love_factors.get(chunk_start + 3).copied().unwrap_or(1.0),
            ]);

            // Antifragility = positive convexity to volatility
            // Simplified: return / volatility * love_factor
            let base_antifragility = rets / vols.max(f64x4::splat(0.001));
            let adjusted = base_antifragility * factors;

            // Normalize to [0, 1]
            let normalized = (adjusted + f64x4::splat(1.0)) / f64x4::splat(2.0);
            let clamped = normalized.max(f64x4::splat(0.0)).min(f64x4::splat(1.0));

            for i in 0..chunk_size {
                results.push(clamped.as_array_ref()[i]);
            }
        }

        Ok(results)
    }

    /// Parallel processing for large datasets
    #[cfg(feature = "parallel")]
    pub fn parallel_risk_assessment<F>(
        data_chunks: &[Vec<f64>],
        processor: F,
    ) -> Result<Vec<f64>, TalebianRiskError>
    where
        F: Fn(&[f64]) -> Result<f64, TalebianRiskError> + Sync + Send,
    {
        data_chunks
            .par_iter()
            .map(|chunk| processor(chunk))
            .collect::<Result<Vec<_>, _>>()
    }

    /// Memory-aligned allocation for SIMD operations
    #[cfg(feature = "aligned-vec")]
    pub fn aligned_allocation(size: usize) -> aligned_vec::AVec<f64> {
        aligned_vec::AVec::with_capacity(32, size) // 32-byte alignment for AVX2
    }
}

/// High-frequency calculation cache
pub struct CalculationCache {
    kelly_cache: std::collections::HashMap<u64, f64>,
    volatility_cache: std::collections::HashMap<u64, f64>,
    whale_cache: std::collections::HashMap<u64, f64>,
    antifragility_cache: std::collections::HashMap<u64, f64>,
    max_entries: usize,
    hits: u64,
    misses: u64,
}

impl CalculationCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            kelly_cache: std::collections::HashMap::with_capacity(max_entries),
            volatility_cache: std::collections::HashMap::with_capacity(max_entries),
            whale_cache: std::collections::HashMap::with_capacity(max_entries),
            antifragility_cache: std::collections::HashMap::with_capacity(max_entries),
            max_entries,
            hits: 0,
            misses: 0,
        }
    }

    pub fn get_kelly(&mut self, key: u64) -> Option<f64> {
        if let Some(value) = self.kelly_cache.get(&key) {
            self.hits += 1;
            Some(*value)
        } else {
            self.misses += 1;
            None
        }
    }

    pub fn set_kelly(&mut self, key: u64, value: f64) {
        if self.kelly_cache.len() >= self.max_entries {
            // Simple eviction: remove oldest entry
            if let Some(first_key) = self.kelly_cache.keys().next().copied() {
                self.kelly_cache.remove(&first_key);
            }
        }
        self.kelly_cache.insert(key, value);
    }

    pub fn get_cache_stats(&self) -> (f64, u64, u64) {
        let hit_rate = if self.hits + self.misses > 0 {
            self.hits as f64 / (self.hits + self.misses) as f64
        } else {
            0.0
        };
        (hit_rate, self.hits, self.misses)
    }

    pub fn clear(&mut self) {
        self.kelly_cache.clear();
        self.volatility_cache.clear();
        self.whale_cache.clear();
        self.antifragility_cache.clear();
        self.hits = 0;
        self.misses = 0;
    }
}

/// Lock-free concurrent data structures for high-frequency updates
#[cfg(feature = "lockfree")]
pub struct LockFreeMetrics {
    calculations_per_second: lockfree::queue::Queue<f64>,
    latency_measurements: lockfree::queue::Queue<u64>,
    error_counts: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "lockfree")]
impl LockFreeMetrics {
    pub fn new() -> Self {
        Self {
            calculations_per_second: lockfree::queue::Queue::new(),
            latency_measurements: lockfree::queue::Queue::new(),
            error_counts: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub fn record_calculation_rate(&self, rate: f64) {
        self.calculations_per_second.push(rate);

        // Keep only recent measurements
        while self.calculations_per_second.len() > 1000 {
            self.calculations_per_second.pop();
        }
    }

    pub fn record_latency(&self, latency_ns: u64) {
        self.latency_measurements.push(latency_ns);

        // Keep only recent measurements
        while self.latency_measurements.len() > 1000 {
            self.latency_measurements.pop();
        }
    }

    pub fn increment_error_count(&self) {
        self.error_counts
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn get_average_rate(&self) -> f64 {
        let rates: Vec<f64> = self.calculations_per_second.iter().collect();
        if rates.is_empty() {
            0.0
        } else {
            rates.iter().sum::<f64>() / rates.len() as f64
        }
    }

    pub fn get_average_latency(&self) -> u64 {
        let latencies: Vec<u64> = self.latency_measurements.iter().collect();
        if latencies.is_empty() {
            0
        } else {
            latencies.iter().sum::<u64>() / latencies.len() as u64
        }
    }
}

/// Memory pool for reducing allocation overhead
#[cfg(feature = "memory-pool")]
pub struct MemoryPool {
    small_buffers: lockfree::queue::Queue<Vec<f64>>,
    medium_buffers: lockfree::queue::Queue<Vec<f64>>,
    large_buffers: lockfree::queue::Queue<Vec<f64>>,
    allocations: std::sync::atomic::AtomicU64,
    deallocations: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "memory-pool")]
impl MemoryPool {
    pub fn new() -> Self {
        let mut pool = Self {
            small_buffers: lockfree::queue::Queue::new(),
            medium_buffers: lockfree::queue::Queue::new(),
            large_buffers: lockfree::queue::Queue::new(),
            allocations: std::sync::atomic::AtomicU64::new(0),
            deallocations: std::sync::atomic::AtomicU64::new(0),
        };

        // Pre-allocate buffers
        for _ in 0..100 {
            pool.small_buffers.push(Vec::with_capacity(64)); // Small: 64 elements
            pool.medium_buffers.push(Vec::with_capacity(256)); // Medium: 256 elements
            pool.large_buffers.push(Vec::with_capacity(1024)); // Large: 1024 elements
        }

        pool
    }

    pub fn get_buffer(&self, size: usize) -> Vec<f64> {
        self.allocations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if size <= 64 {
            self.small_buffers
                .pop()
                .unwrap_or_else(|| Vec::with_capacity(64))
        } else if size <= 256 {
            self.medium_buffers
                .pop()
                .unwrap_or_else(|| Vec::with_capacity(256))
        } else if size <= 1024 {
            self.large_buffers
                .pop()
                .unwrap_or_else(|| Vec::with_capacity(1024))
        } else {
            Vec::with_capacity(size)
        }
    }

    pub fn return_buffer(&self, mut buffer: Vec<f64>) {
        self.deallocations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        buffer.clear();
        let capacity = buffer.capacity();

        if capacity <= 64 && self.small_buffers.len() < 100 {
            self.small_buffers.push(buffer);
        } else if capacity <= 256 && self.medium_buffers.len() < 100 {
            self.medium_buffers.push(buffer);
        } else if capacity <= 1024 && self.large_buffers.len() < 100 {
            self.large_buffers.push(buffer);
        }
        // Drop large buffers to avoid memory bloat
    }

    pub fn get_stats(&self) -> (u64, u64, usize, usize, usize) {
        let allocations = self.allocations.load(std::sync::atomic::Ordering::Relaxed);
        let deallocations = self
            .deallocations
            .load(std::sync::atomic::Ordering::Relaxed);
        (
            allocations,
            deallocations,
            self.small_buffers.len(),
            self.medium_buffers.len(),
            self.large_buffers.len(),
        )
    }
}

/// CPU feature detection for optimal SIMD usage
pub struct CpuFeatures;

impl CpuFeatures {
    /// Detect available SIMD features
    pub fn detect() -> SimdCapabilities {
        SimdCapabilities {
            sse2: Self::has_sse2(),
            sse4_1: Self::has_sse4_1(),
            avx: Self::has_avx(),
            avx2: Self::has_avx2(),
            avx512f: Self::has_avx512f(),
            neon: Self::has_neon(),
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn has_sse2() -> bool {
        is_x86_feature_detected!("sse2")
    }

    #[cfg(target_arch = "x86_64")]
    fn has_sse4_1() -> bool {
        is_x86_feature_detected!("sse4.1")
    }

    #[cfg(target_arch = "x86_64")]
    fn has_avx() -> bool {
        is_x86_feature_detected!("avx")
    }

    #[cfg(target_arch = "x86_64")]
    fn has_avx2() -> bool {
        is_x86_feature_detected!("avx2")
    }

    #[cfg(target_arch = "x86_64")]
    fn has_avx512f() -> bool {
        is_x86_feature_detected!("avx512f")
    }

    #[cfg(target_arch = "x86_64")]
    fn has_neon() -> bool {
        false // NEON not available on x86_64
    }

    #[cfg(target_arch = "aarch64")]
    fn has_neon() -> bool {
        std::arch::is_aarch64_feature_detected!("neon")
    }

    // Fallbacks for non-x86/ARM architectures
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn has_sse2() -> bool {
        false
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn has_sse4_1() -> bool {
        false
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn has_avx() -> bool {
        false
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn has_avx2() -> bool {
        false
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn has_avx512f() -> bool {
        false
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn has_neon() -> bool {
        false
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimdCapabilities {
    pub sse2: bool,
    pub sse4_1: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub neon: bool,
}

impl SimdCapabilities {
    pub fn best_vector_width(&self) -> usize {
        if self.avx512f {
            8 // 512-bit / 64-bit = 8 f64 elements
        } else if self.avx2 || self.avx {
            4 // 256-bit / 64-bit = 4 f64 elements
        } else if self.sse2 {
            2 // 128-bit / 64-bit = 2 f64 elements
        } else {
            1 // Scalar fallback
        }
    }

    pub fn optimal_chunk_size(&self) -> usize {
        self.best_vector_width() * 16 // Process multiple vectors per chunk
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::new();

        // Record some calculations
        monitor.record_calculation(1000); // 1μs
        monitor.record_calculation(2000); // 2μs
        monitor.record_cache_hit();
        monitor.record_cache_miss();

        let stats = monitor.get_stats();
        assert_eq!(stats.average_latency_ns, 1500);
        assert_eq!(stats.peak_latency_ns, 2000);
        assert_eq!(stats.cache_hit_rate, 0.5);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_kelly_calculation() {
        let expected_returns = vec![0.01, 0.02, 0.015, 0.025];
        let variances = vec![0.001, 0.002, 0.0015, 0.0025];
        let whale_multipliers = vec![1.0, 1.5, 1.2, 1.3];

        let results =
            SimdMath::kelly_fraction_simd_x4(&expected_returns, &variances, &whale_multipliers)
                .unwrap();

        assert_eq!(results.len(), 4);
        for &result in &results {
            assert!(result >= 0.05);
            assert!(result <= 0.75);
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_volatility_calculation() {
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015, -0.008, 0.012, 0.003];

        let volatility = SimdMath::volatility_simd_x8(&returns).unwrap();

        assert!(volatility > 0.0);
        assert!(volatility < 1.0); // Reasonable volatility range
    }

    #[test]
    fn test_calculation_cache() {
        let mut cache = CalculationCache::new(10);

        // Test cache miss
        assert!(cache.get_kelly(123).is_none());

        // Test cache set and hit
        cache.set_kelly(123, 0.5);
        assert_eq!(cache.get_kelly(123), Some(0.5));

        let (hit_rate, hits, misses) = cache.get_cache_stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
        assert_eq!(hit_rate, 0.5);
    }

    #[test]
    fn test_cpu_feature_detection() {
        let capabilities = CpuFeatures::detect();

        // Basic sanity checks
        let vector_width = capabilities.best_vector_width();
        assert!(vector_width >= 1);
        assert!(vector_width <= 8);

        let chunk_size = capabilities.optimal_chunk_size();
        assert!(chunk_size >= 16);
        assert!(chunk_size <= 128);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_whale_scoring() {
        let volume_ratios = vec![2.0, 1.5, 3.0, 1.8];
        let price_impacts = vec![0.02, 0.015, 0.03, 0.018];
        let order_imbalances = vec![0.7, 0.6, 0.8, 0.65];
        let smart_money_flows = vec![0.8, 0.7, 0.9, 0.75];

        let scores = SimdMath::whale_score_simd_x4(
            &volume_ratios,
            &price_impacts,
            &order_imbalances,
            &smart_money_flows,
        )
        .unwrap();

        assert_eq!(scores.len(), 4);
        for &score in &scores {
            assert!(score >= 0.0);
            assert!(score <= 1.0);
        }

        // Highest inputs should produce highest score
        let max_score_idx = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        assert_eq!(max_score_idx, 2); // Third element has highest inputs
    }

    #[test]
    #[cfg(feature = "memory-pool")]
    fn test_memory_pool() {
        let pool = MemoryPool::new();

        // Get and return buffers
        let buffer1 = pool.get_buffer(50); // Should get small buffer
        let buffer2 = pool.get_buffer(200); // Should get medium buffer
        let buffer3 = pool.get_buffer(800); // Should get large buffer

        pool.return_buffer(buffer1);
        pool.return_buffer(buffer2);
        pool.return_buffer(buffer3);

        let (allocations, deallocations, small, medium, large) = pool.get_stats();
        assert_eq!(allocations, 3);
        assert_eq!(deallocations, 3);
        assert!(small > 0);
        assert!(medium > 0);
        assert!(large > 0);
    }
}
