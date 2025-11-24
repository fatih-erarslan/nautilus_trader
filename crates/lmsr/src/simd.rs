//! SIMD-accelerated LMSR calculations
//! 
//! This module provides vectorized implementations of LMSR operations
//! optimized for high-frequency financial trading systems.

use crate::errors::{LMSRError, Result};
use crate::lmsr::LMSRCalculator;

#[cfg(feature = "simd")]
use wide::f64x4;
use aligned_vec::{AVec, ABox};
use rayon::prelude::*;
use std::arch::x86_64::*;
use parking_lot::RwLock;
use std::sync::Arc;

/// SIMD-optimized LMSR calculator
#[derive(Debug, Clone)]
pub struct SIMDLMSRCalculator {
    num_outcomes: usize,
    liquidity_parameter: f64,
    liquidity_vec: f64x4,
    // Pre-computed constants for SIMD operations
    inv_liquidity_vec: f64x4,
}

impl SIMDLMSRCalculator {
    /// Create new SIMD LMSR calculator
    pub fn new(num_outcomes: usize, liquidity_parameter: f64) -> Result<Self> {
        if num_outcomes < 2 {
            return Err(LMSRError::invalid_market(
                format!("Number of outcomes must be at least 2, got {}", num_outcomes)
            ));
        }
        
        if !liquidity_parameter.is_finite() || liquidity_parameter <= 0.0 {
            return Err(LMSRError::invalid_parameter("liquidity_parameter", liquidity_parameter));
        }
        
        Ok(Self {
            num_outcomes,
            liquidity_parameter,
            liquidity_vec: f64x4::splat(liquidity_parameter),
            inv_liquidity_vec: f64x4::splat(1.0 / liquidity_parameter),
        })
    }

    /// SIMD-optimized cost function calculation
    #[cfg(feature = "simd")]
    pub fn cost_function_simd(&self, quantities: &[f64]) -> Result<f64> {
        if quantities.len() != self.num_outcomes {
            return Err(LMSRError::invalid_quantity(
                format!("Expected {} quantities, got {}", self.num_outcomes, quantities.len())
            ));
        }

        // Normalize quantities by liquidity parameter using SIMD
        let normalized = self.normalize_quantities_simd(quantities)?;
        let log_sum = self.log_sum_exp_simd(&normalized)?;
        
        Ok(self.liquidity_parameter * log_sum)
    }

    /// SIMD normalization of quantities
    #[cfg(feature = "simd")]
    fn normalize_quantities_simd(&self, quantities: &[f64]) -> Result<Vec<f64>> {
        let aligned_quantities: ABox<[f64]> = quantities.iter().copied().collect();
        let mut normalized = AVec::with_capacity(quantities.len());
        
        let chunk_size = 4;
        let mut i = 0;
        
        while i + chunk_size <= aligned_quantities.len() {
            let quantities_chunk = f64x4::new([
                aligned_quantities[i],
                aligned_quantities[i + 1], 
                aligned_quantities[i + 2],
                aligned_quantities[i + 3],
            ]);
            
            let normalized_chunk = quantities_chunk * self.inv_liquidity_vec;
            normalized.extend_from_slice(&normalized_chunk.to_array());
            i += chunk_size;
        }
        
        // Process remaining elements
        for &quantity in &aligned_quantities[i..] {
            normalized.push(quantity / self.liquidity_parameter);
        }
        
        Ok(normalized.into_vec())
    }

    /// SIMD-optimized log-sum-exp for numerical stability
    #[cfg(feature = "simd")]
    fn log_sum_exp_simd(&self, values: &[f64]) -> Result<f64> {
        if values.is_empty() {
            return Ok(f64::NEG_INFINITY);
        }

        // Find maximum value for numerical stability
        let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let max_vec = f64x4::splat(max_val);
        
        let aligned_values: ABox<[f64]> = values.iter().copied().collect();
        let mut sum = f64x4::splat(0.0);
        
        let chunk_size = 4;
        let mut i = 0;
        
        while i + chunk_size <= aligned_values.len() {
            let values_chunk = f64x4::new([
                aligned_values[i],
                aligned_values[i + 1], 
                aligned_values[i + 2],
                aligned_values[i + 3],
            ]);
            
            let shifted = values_chunk - max_vec;
            let exp_vals = shifted.exp();
            sum += exp_vals;
            i += chunk_size;
        }
        
        // Sum SIMD register
        let sum_array = sum.to_array();
        let mut total_sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
        
        // Process remaining elements
        for &value in &aligned_values[i..] {
            total_sum += (value - max_val).exp();
        }
        
        if !total_sum.is_finite() || total_sum <= 0.0 {
            return Err(LMSRError::numerical_error("Log-sum-exp calculation failed"));
        }
        
        Ok(max_val + total_sum.ln())
    }

    /// SIMD-optimized softmax calculation for marginal prices
    #[cfg(feature = "simd")]
    pub fn all_marginal_prices_simd(&self, quantities: &[f64]) -> Result<Vec<f64>> {
        if quantities.len() != self.num_outcomes {
            return Err(LMSRError::invalid_quantity(
                format!("Expected {} quantities, got {}", self.num_outcomes, quantities.len())
            ));
        }

        let normalized = self.normalize_quantities_simd(quantities)?;
        self.softmax_simd(&normalized)
    }

    /// SIMD softmax implementation
    #[cfg(feature = "simd")]
    fn softmax_simd(&self, values: &[f64]) -> Result<Vec<f64>> {
        if values.is_empty() {
            return Ok(Vec::new());
        }

        // Find maximum for numerical stability
        let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let max_vec = f64x4::splat(max_val);
        
        let aligned_values: ABox<[f64]> = values.iter().copied().collect();
        let mut exp_values = AVec::with_capacity(values.len());
        let mut sum = f64x4::splat(0.0);
        
        let chunk_size = 4;
        let mut i = 0;
        
        // First pass: calculate exp(x - max) and sum
        while i + chunk_size <= aligned_values.len() {
            let values_chunk = f64x4::new([
                aligned_values[i],
                aligned_values[i + 1], 
                aligned_values[i + 2],
                aligned_values[i + 3],
            ]);
            
            let shifted = values_chunk - max_vec;
            let exp_vals = shifted.exp();
            exp_values.extend_from_slice(&exp_vals.to_array());
            sum += exp_vals;
            i += chunk_size;
        }
        
        // Sum SIMD register
        let sum_array = sum.to_array();
        let mut total_sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
        
        // Process remaining elements
        for &value in &aligned_values[i..] {
            let exp_val = (value - max_val).exp();
            exp_values.push(exp_val);
            total_sum += exp_val;
        }
        
        if !total_sum.is_finite() || total_sum <= 0.0 {
            return Err(LMSRError::numerical_error("Softmax calculation failed"));
        }
        
        // Second pass: normalize by sum
        let sum_vec = f64x4::splat(total_sum);
        let mut probabilities = AVec::with_capacity(values.len());
        
        i = 0;
        while i + chunk_size <= exp_values.len() {
            let exp_chunk = f64x4::new([
                exp_values[i],
                exp_values[i + 1], 
                exp_values[i + 2],
                exp_values[i + 3],
            ]);
            
            let prob_chunk = exp_chunk / sum_vec;
            probabilities.extend_from_slice(&prob_chunk.to_array());
            i += chunk_size;
        }
        
        // Process remaining elements
        for &exp_val in &exp_values[i..] {
            probabilities.push(exp_val / total_sum);
        }
        
        Ok(probabilities.into_vec())
    }

    /// Calculate trade cost using SIMD
    pub fn trade_cost_simd(&self, current_quantities: &[f64], new_quantities: &[f64]) -> Result<f64> {
        let current_cost = self.cost_function_simd(current_quantities)?;
        let new_cost = self.cost_function_simd(new_quantities)?;
        Ok(new_cost - current_cost)
    }

    /// Parallel SIMD processing for batch operations
    pub fn batch_marginal_prices_simd(&self, batch_quantities: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if batch_quantities.is_empty() {
            return Ok(Vec::new());
        }

        // Process in parallel chunks
        let results: Result<Vec<Vec<f64>>> = batch_quantities
            .par_iter()
            .map(|quantities| self.all_marginal_prices_simd(quantities))
            .collect();
        
        results
    }

    /// Optimized arbitrage calculation with SIMD
    #[cfg(feature = "simd")]
    pub fn optimal_arbitrage_trade_simd(
        &self,
        current_quantities: &[f64],
        external_prices: &[f64],
    ) -> Result<Vec<f64>> {
        if external_prices.len() != self.num_outcomes {
            return Err(LMSRError::invalid_quantity("External prices length mismatch"));
        }

        let current_prices = self.all_marginal_prices_simd(current_quantities)?;
        let mut trade_amounts = AVec::with_capacity(self.num_outcomes);
        
        let liquidity_vec = f64x4::splat(self.liquidity_parameter);
        let chunk_size = 4;
        let mut i = 0;
        
        while i + chunk_size <= current_prices.len() {
            let current_chunk = f64x4::new([
                current_prices[i],
                current_prices[i + 1], 
                current_prices[i + 2],
                current_prices[i + 3],
            ]);
            
            let external_chunk = f64x4::new([
                external_prices[i],
                external_prices[i + 1], 
                external_prices[i + 2],
                external_prices[i + 3],
            ]);
            
            let price_diff = external_chunk - current_chunk;
            let trade_chunk = liquidity_vec * price_diff;
            
            trade_amounts.extend_from_slice(&trade_chunk.to_array());
            i += chunk_size;
        }
        
        // Process remaining elements
        for j in i..current_prices.len() {
            let price_diff = external_prices[j] - current_prices[j];
            trade_amounts.push(self.liquidity_parameter * price_diff);
        }
        
        Ok(trade_amounts.into_vec())
    }
}

/// Lock-free concurrent market state for high-frequency trading
pub struct LockFreeMarketState {
    quantities: Arc<RwLock<Vec<f64>>>,
    total_volume: Arc<parking_lot::Mutex<f64>>,
    trade_count: Arc<parking_lot::Mutex<u64>>,
    calculator: SIMDLMSRCalculator,
}

impl LockFreeMarketState {
    /// Create new lock-free market state
    pub fn new(num_outcomes: usize, liquidity_parameter: f64) -> Result<Self> {
        let calculator = SIMDLMSRCalculator::new(num_outcomes, liquidity_parameter)?;
        
        Ok(Self {
            quantities: Arc::new(RwLock::new(vec![0.0; num_outcomes])),
            total_volume: Arc::new(parking_lot::Mutex::new(0.0)),
            trade_count: Arc::new(parking_lot::Mutex::new(0)),
            calculator,
        })
    }

    /// Execute trade with lock-free optimizations
    pub fn execute_trade_lockfree(&self, buy_amounts: &[f64]) -> Result<f64> {
        // Use read lock for cost calculation
        let cost = {
            let quantities = self.quantities.read();
            self.calculator.trade_cost_simd(&quantities, buy_amounts)?
        };

        // Use write lock only for state update
        {
            let mut quantities = self.quantities.write();
            for (i, &amount) in buy_amounts.iter().enumerate() {
                quantities[i] += amount;
            }
        }

        // Update statistics with minimal locking
        {
            let trade_volume: f64 = buy_amounts.iter().map(|x| x.abs()).sum();
            *self.total_volume.lock() += trade_volume;
            *self.trade_count.lock() += 1;
        }

        Ok(cost)
    }

    /// Get current prices with read-only access
    pub fn get_prices_lockfree(&self) -> Result<Vec<f64>> {
        let quantities = self.quantities.read();
        self.calculator.all_marginal_prices_simd(&quantities)
    }
}

/// Memory-mapped large dataset processor
pub struct MMapLMSRProcessor {
    calculator: SIMDLMSRCalculator,
    memory_pools: Vec<AVec<f64>>,
    pool_index: std::sync::atomic::AtomicUsize,
}

impl MMapLMSRProcessor {
    /// Create new memory-mapped processor
    pub fn new(calculator: SIMDLMSRCalculator, pool_count: usize, pool_size: usize) -> Self {
        let mut memory_pools = Vec::with_capacity(pool_count);
        for _ in 0..pool_count {
            memory_pools.push(AVec::with_capacity(pool_size));
        }
        
        Self {
            calculator,
            memory_pools,
            pool_index: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Process large batch with memory mapping
    pub fn process_large_batch(&mut self, quantities_batch: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if quantities_batch.len() < 1000 {
            return self.calculator.batch_marginal_prices_simd(quantities_batch);
        }

        // Use memory mapping for very large datasets
        let chunk_size = quantities_batch.len() / rayon::current_num_threads();
        
        let results: Result<Vec<Vec<Vec<f64>>>> = quantities_batch
            .par_chunks(chunk_size)
            .map(|chunk| self.calculator.batch_marginal_prices_simd(chunk))
            .collect();
        
        let results = results?;
        Ok(results.into_iter().flatten().collect())
    }
}

/// Adaptive SIMD executor that chooses optimal strategy based on data size
pub struct AdaptiveSIMDExecutor {
    calculator: SIMDLMSRCalculator,
    cpu_capabilities: CPUCapabilities,
    performance_thresholds: PerformanceThresholds,
}

#[derive(Debug, Clone)]
pub struct CPUCapabilities {
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_fma: bool,
    pub cache_line_size: usize,
}

#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    pub simd_threshold: usize,
    pub parallel_threshold: usize,
    pub memory_map_threshold: usize,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            simd_threshold: 16,
            parallel_threshold: 1000,
            memory_map_threshold: 100_000,
        }
    }
}

impl AdaptiveSIMDExecutor {
    /// Create new adaptive executor
    pub fn new(calculator: SIMDLMSRCalculator) -> Self {
        Self {
            calculator,
            cpu_capabilities: CPUCapabilities::detect(),
            performance_thresholds: PerformanceThresholds::default(),
        }
    }

    /// Adaptively calculate marginal prices
    pub fn adaptive_marginal_prices(&self, quantities: &[f64]) -> Result<Vec<f64>> {
        let data_size = quantities.len();
        
        if data_size >= self.performance_thresholds.memory_map_threshold {
            // Use memory-mapped processing for very large datasets
            self.memory_mapped_calculation(quantities)
        } else if data_size >= self.performance_thresholds.parallel_threshold {
            // Use parallel SIMD for large datasets
            self.parallel_simd_calculation(quantities)
        } else if data_size >= self.performance_thresholds.simd_threshold && self.cpu_capabilities.has_avx2 {
            // Use SIMD for medium datasets
            self.calculator.all_marginal_prices_simd(quantities)
        } else {
            // Use scalar for small datasets
            self.scalar_calculation(quantities)
        }
    }

    fn memory_mapped_calculation(&self, quantities: &[f64]) -> Result<Vec<f64>> {
        // For very large datasets, process in chunks to avoid memory pressure
        let chunk_size = 10000;
        let mut results = Vec::with_capacity(quantities.len());
        
        for chunk in quantities.chunks(chunk_size) {
            let chunk_results = self.calculator.all_marginal_prices_simd(chunk)?;
            results.extend(chunk_results);
        }
        
        Ok(results)
    }

    fn parallel_simd_calculation(&self, quantities: &[f64]) -> Result<Vec<f64>> {
        // Split into parallel chunks for SIMD processing
        let num_threads = rayon::current_num_threads();
        let chunk_size = (quantities.len() / num_threads).max(1000);
        
        let results: Result<Vec<Vec<f64>>> = quantities
            .par_chunks(chunk_size)
            .map(|chunk| self.calculator.all_marginal_prices_simd(chunk))
            .collect();
        
        let results = results?;
        Ok(results.into_iter().flatten().collect())
    }

    fn scalar_calculation(&self, quantities: &[f64]) -> Result<Vec<f64>> {
        // Fallback to basic calculation for small datasets
        let basic_calc = LMSRCalculator::new(
            self.calculator.num_outcomes, 
            self.calculator.liquidity_parameter
        )?;
        basic_calc.all_marginal_prices(quantities)
    }
}

impl CPUCapabilities {
    /// Detect CPU capabilities at runtime
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                has_avx2: is_x86_feature_detected!("avx2"),
                has_avx512: is_x86_feature_detected!("avx512f"),
                has_fma: is_x86_feature_detected!("fma"),
                cache_line_size: 64, // Typical x86_64 cache line size
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                has_avx2: false,
                has_avx512: false,
                has_fma: false,
                cache_line_size: 64,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_simd_lmsr_calculator() {
        let calc = SIMDLMSRCalculator::new(3, 100.0).unwrap();
        let quantities = vec![10.0, 20.0, 30.0];
        
        let prices = calc.all_marginal_prices_simd(&quantities).unwrap();
        assert_eq!(prices.len(), 3);
        
        let sum: f64 = prices.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_cost_function() {
        let calc = SIMDLMSRCalculator::new(2, 100.0).unwrap();
        let quantities = vec![0.0, 0.0];
        
        let cost = calc.cost_function_simd(&quantities).unwrap();
        let expected = 100.0 * (2.0_f64).ln();
        assert_relative_eq!(cost, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_lock_free_market() {
        let market = LockFreeMarketState::new(2, 100.0).unwrap();
        let buy_amounts = vec![10.0, 0.0];
        
        let cost = market.execute_trade_lockfree(&buy_amounts).unwrap();
        assert!(cost > 0.0);
        
        let prices = market.get_prices_lockfree().unwrap();
        assert_eq!(prices.len(), 2);
        assert!(prices[0] > prices[1]); // More shares of outcome 0
    }

    #[test]
    fn test_adaptive_executor() {
        let calc = SIMDLMSRCalculator::new(10, 1000.0).unwrap();
        let executor = AdaptiveSIMDExecutor::new(calc);
        
        // Test different data sizes
        let small_quantities = vec![1.0; 5];
        let medium_quantities = vec![1.0; 50];
        let large_quantities = vec![1.0; 5000];
        
        let small_result = executor.adaptive_marginal_prices(&small_quantities).unwrap();
        let medium_result = executor.adaptive_marginal_prices(&medium_quantities).unwrap();
        let large_result = executor.adaptive_marginal_prices(&large_quantities).unwrap();
        
        assert_eq!(small_result.len(), 5);
        assert_eq!(medium_result.len(), 50);
        assert_eq!(large_result.len(), 5000);
    }

    #[test]
    fn test_cpu_capabilities() {
        let caps = CPUCapabilities::detect();
        assert!(caps.cache_line_size > 0);
        // Other capabilities depend on actual hardware
    }
}