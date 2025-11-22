//! SIMD-accelerated prospect theory calculations
//! 
//! This module provides vectorized implementations of core prospect theory
//! operations using SIMD instructions for maximum performance in financial trading.

use crate::errors::{ProspectTheoryError, Result};
use crate::value_function::ValueFunctionParams;
use crate::probability_weighting::WeightingParams;

#[cfg(feature = "simd")]
use wide::f64x4;
use aligned_vec::{AVec, ABox};
use rayon::prelude::*;
use std::arch::x86_64::*;

/// SIMD-optimized value function calculator
#[derive(Debug, Clone)]
pub struct SIMDValueFunction {
    params: ValueFunctionParams,
    // Pre-computed constants for SIMD operations
    alpha_vec: f64x4,
    beta_vec: f64x4,
    lambda_vec: f64x4,
    reference_vec: f64x4,
}

impl SIMDValueFunction {
    /// Create new SIMD value function
    pub fn new(params: ValueFunctionParams) -> Result<Self> {
        params.validate()?;
        
        Ok(Self {
            alpha_vec: f64x4::splat(params.alpha),
            beta_vec: f64x4::splat(params.beta),
            lambda_vec: f64x4::splat(params.lambda),
            reference_vec: f64x4::splat(params.reference_point),
            params,
        })
    }

    /// Calculate value function for multiple outcomes using SIMD
    #[cfg(feature = "simd")]
    pub fn values_simd(&self, outcomes: &[f64]) -> Result<Vec<f64>> {
        if outcomes.is_empty() {
            return Ok(Vec::new());
        }

        // Ensure data is aligned for SIMD operations
        let aligned_outcomes: ABox<[f64]> = outcomes.iter().copied().collect();
        let mut aligned_results = AVec::with_capacity(outcomes.len());
        
        // Process in chunks of 4 (f64x4)
        let chunk_size = 4;
        let mut i = 0;
        
        while i + chunk_size <= aligned_outcomes.len() {
            let outcomes_chunk = f64x4::new([
                aligned_outcomes[i],
                aligned_outcomes[i + 1], 
                aligned_outcomes[i + 2],
                aligned_outcomes[i + 3],
            ]);
            
            let result_chunk = self.value_function_simd(outcomes_chunk);
            aligned_results.extend_from_slice(&result_chunk.to_array());
            i += chunk_size;
        }
        
        // Process remaining elements
        for &outcome in &aligned_outcomes[i..] {
            aligned_results.push(self.value_scalar(outcome)?);
        }
        
        Ok(aligned_results.into_vec())
    }

    /// SIMD value function implementation for 4 values at once
    #[cfg(feature = "simd")]
    fn value_function_simd(&self, outcomes: f64x4) -> f64x4 {
        let relative_outcomes = outcomes - self.reference_vec;
        
        // Create masks for gains and losses
        let zero_vec = f64x4::splat(0.0);
        let gains_mask = relative_outcomes.cmp_ge(zero_vec);
        let losses_mask = !gains_mask;
        
        // Calculate gains: x^α
        let gains_result = self.calculate_gains_simd(relative_outcomes);
        
        // Calculate losses: -λ * |x|^β  
        let losses_result = self.calculate_losses_simd(relative_outcomes);
        
        // Blend results based on masks
        gains_mask.blend(gains_result, losses_result)
    }

    /// SIMD gains calculation
    #[cfg(feature = "simd")]
    fn calculate_gains_simd(&self, relative_outcomes: f64x4) -> f64x4 {
        // For gains: V(x) = x^α
        let zero_vec = f64x4::splat(0.0);
        let epsilon_vec = f64x4::splat(f64::EPSILON);
        
        // Handle zero case
        let is_zero = relative_outcomes.cmp_lt(epsilon_vec);
        
        // Power calculation using SIMD-friendly approach
        let powered = self.fast_pow_simd(relative_outcomes, self.alpha_vec);
        
        is_zero.blend(zero_vec, powered)
    }

    /// SIMD losses calculation  
    #[cfg(feature = "simd")]
    fn calculate_losses_simd(&self, relative_outcomes: f64x4) -> f64x4 {
        // For losses: V(x) = -λ * |x|^β
        let abs_loss = relative_outcomes.abs();
        let powered = self.fast_pow_simd(abs_loss, self.beta_vec);
        -self.lambda_vec * powered
    }

    /// Fast SIMD power function approximation
    #[cfg(feature = "simd")]
    fn fast_pow_simd(&self, base: f64x4, exponent: f64x4) -> f64x4 {
        // Use exp(exponent * ln(base)) for SIMD-friendly power calculation
        let ln_base = base.ln();
        let product = exponent * ln_base;
        product.exp()
    }

    /// Fallback scalar implementation
    fn value_scalar(&self, outcome: f64) -> Result<f64> {
        let relative_outcome = outcome - self.params.reference_point;
        
        if relative_outcome >= 0.0 {
            if relative_outcome.abs() < f64::EPSILON {
                Ok(0.0)
            } else {
                Ok(relative_outcome.powf(self.params.alpha))
            }
        } else {
            let abs_loss = relative_outcome.abs();
            Ok(-self.params.lambda * abs_loss.powf(self.params.beta))
        }
    }

    /// Parallel SIMD processing for large datasets
    pub fn values_parallel_simd(&self, outcomes: &[f64]) -> Result<Vec<f64>> {
        if outcomes.len() < 1000 {
            return self.values_simd(outcomes);
        }

        // Split into chunks for parallel processing
        let chunk_size = (outcomes.len() / rayon::current_num_threads()).max(1000);
        
        let results: Result<Vec<Vec<f64>>> = outcomes
            .par_chunks(chunk_size)
            .map(|chunk| self.values_simd(chunk))
            .collect();
        
        let results = results?;
        Ok(results.into_iter().flatten().collect())
    }
}

/// SIMD-optimized probability weighting
#[derive(Debug, Clone)]
pub struct SIMDProbabilityWeighting {
    params: WeightingParams,
    gamma_gains_vec: f64x4,
    gamma_losses_vec: f64x4,
    delta_gains_vec: f64x4,
    delta_losses_vec: f64x4,
}

impl SIMDProbabilityWeighting {
    /// Create new SIMD probability weighting
    pub fn new(params: WeightingParams) -> Result<Self> {
        params.validate()?;
        
        Ok(Self {
            gamma_gains_vec: f64x4::splat(params.gamma_gains),
            gamma_losses_vec: f64x4::splat(params.gamma_losses),
            delta_gains_vec: f64x4::splat(params.delta_gains),
            delta_losses_vec: f64x4::splat(params.delta_losses),
            params,
        })
    }

    /// Calculate weights for gains using SIMD
    #[cfg(feature = "simd")]
    pub fn weights_gains_simd(&self, probabilities: &[f64]) -> Result<Vec<f64>> {
        if probabilities.is_empty() {
            return Ok(Vec::new());
        }

        let aligned_probs: ABox<[f64]> = probabilities.iter().copied().collect();
        let mut aligned_results = AVec::with_capacity(probabilities.len());
        
        let chunk_size = 4;
        let mut i = 0;
        
        while i + chunk_size <= aligned_probs.len() {
            let probs_chunk = f64x4::new([
                aligned_probs[i],
                aligned_probs[i + 1], 
                aligned_probs[i + 2],
                aligned_probs[i + 3],
            ]);
            
            let result_chunk = self.tversky_kahneman_gains_simd(probs_chunk);
            aligned_results.extend_from_slice(&result_chunk.to_array());
            i += chunk_size;
        }
        
        // Process remaining elements
        for &prob in &aligned_probs[i..] {
            aligned_results.push(self.weight_gains_scalar(prob)?);
        }
        
        Ok(aligned_results.into_vec())
    }

    /// SIMD Tversky-Kahneman weighting for gains
    #[cfg(feature = "simd")]
    fn tversky_kahneman_gains_simd(&self, probs: f64x4) -> f64x4 {
        let zero_vec = f64x4::splat(0.0);
        let one_vec = f64x4::splat(1.0);
        
        // Handle boundary conditions
        let is_zero = probs.cmp_le(zero_vec);
        let is_one = probs.cmp_ge(one_vec);
        
        // Calculate: δ * p^γ / (δ * p^γ + (1-p)^γ)
        let p_gamma = self.fast_pow_simd(probs, self.gamma_gains_vec);
        let one_minus_p = one_vec - probs;
        let one_minus_p_gamma = self.fast_pow_simd(one_minus_p, self.gamma_gains_vec);
        
        let numerator = self.delta_gains_vec * p_gamma;
        let denominator = numerator + one_minus_p_gamma;
        let result = numerator / denominator;
        
        // Apply boundary conditions
        let result = is_zero.blend(zero_vec, result);
        let result = is_one.blend(one_vec, result);
        
        result
    }

    /// Fast power function for SIMD
    #[cfg(feature = "simd")]
    fn fast_pow_simd(&self, base: f64x4, exponent: f64x4) -> f64x4 {
        let ln_base = base.ln();
        let product = exponent * ln_base;
        product.exp()
    }

    /// Scalar fallback
    fn weight_gains_scalar(&self, probability: f64) -> Result<f64> {
        if probability <= 0.0 {
            return Ok(0.0);
        }
        if probability >= 1.0 {
            return Ok(1.0);
        }

        let p_gamma = probability.powf(self.params.gamma_gains);
        let one_minus_p = 1.0 - probability;
        let one_minus_p_gamma = one_minus_p.powf(self.params.gamma_gains);
        
        let numerator = self.params.delta_gains * p_gamma;
        let denominator = numerator + one_minus_p_gamma;
        
        Ok(numerator / denominator)
    }

    /// Parallel SIMD processing for large datasets
    pub fn weights_gains_parallel_simd(&self, probabilities: &[f64]) -> Result<Vec<f64>> {
        if probabilities.len() < 1000 {
            return self.weights_gains_simd(probabilities);
        }

        let chunk_size = (probabilities.len() / rayon::current_num_threads()).max(1000);
        
        let results: Result<Vec<Vec<f64>>> = probabilities
            .par_chunks(chunk_size)
            .map(|chunk| self.weights_gains_simd(chunk))
            .collect();
        
        let results = results?;
        Ok(results.into_iter().flatten().collect())
    }
}

/// CPU capability detection for adaptive SIMD
pub struct CPUCapabilities {
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_fma: bool,
}

impl CPUCapabilities {
    /// Detect CPU SIMD capabilities at runtime
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                has_avx2: is_x86_feature_detected!("avx2"),
                has_avx512: is_x86_feature_detected!("avx512f"),
                has_fma: is_x86_feature_detected!("fma"),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                has_avx2: false,
                has_avx512: false,
                has_fma: false,
            }
        }
    }
}

/// Memory pool for high-frequency allocations
pub struct MemoryPool {
    value_pools: Vec<AVec<f64>>,
    weight_pools: Vec<AVec<f64>>,
    pool_size: usize,
}

impl MemoryPool {
    /// Create new memory pool
    pub fn new(pool_size: usize, initial_capacity: usize) -> Self {
        let mut value_pools = Vec::with_capacity(pool_size);
        let mut weight_pools = Vec::with_capacity(pool_size);
        
        for _ in 0..pool_size {
            value_pools.push(AVec::with_capacity(initial_capacity));
            weight_pools.push(AVec::with_capacity(initial_capacity));
        }
        
        Self {
            value_pools,
            weight_pools,
            pool_size,
        }
    }

    /// Get aligned vector for values
    pub fn get_value_vec(&mut self, size: usize) -> AVec<f64> {
        if let Some(mut vec) = self.value_pools.pop() {
            vec.clear();
            vec.reserve(size);
            vec
        } else {
            AVec::with_capacity(size)
        }
    }

    /// Return vector to pool
    pub fn return_value_vec(&mut self, vec: AVec<f64>) {
        if self.value_pools.len() < self.pool_size {
            self.value_pools.push(vec);
        }
    }
}

/// High-performance prospect theory calculator with all optimizations
pub struct HighPerformanceProspectCalculator {
    simd_value_fn: SIMDValueFunction,
    simd_weighting: SIMDProbabilityWeighting,
    memory_pool: MemoryPool,
    cpu_caps: CPUCapabilities,
}

impl HighPerformanceProspectCalculator {
    /// Create new high-performance calculator
    pub fn new(
        value_params: ValueFunctionParams,
        weighting_params: WeightingParams,
    ) -> Result<Self> {
        Ok(Self {
            simd_value_fn: SIMDValueFunction::new(value_params)?,
            simd_weighting: SIMDProbabilityWeighting::new(weighting_params)?,
            memory_pool: MemoryPool::new(16, 10000),
            cpu_caps: CPUCapabilities::detect(),
        })
    }

    /// Calculate prospect value with maximum performance
    pub fn calculate_prospect_value(
        &mut self,
        outcomes: &[f64],
        probabilities: &[f64],
    ) -> Result<f64> {
        if outcomes.len() != probabilities.len() {
            return Err(ProspectTheoryError::computation_failed(
                "Outcomes and probabilities must have same length",
            ));
        }

        // Use SIMD for calculation
        let values = if self.cpu_caps.has_avx2 && outcomes.len() >= 16 {
            self.simd_value_fn.values_parallel_simd(outcomes)?
        } else {
            self.simd_value_fn.values_simd(outcomes)?
        };

        let decision_weights = if self.cpu_caps.has_avx2 && probabilities.len() >= 16 {
            self.simd_weighting.weights_gains_parallel_simd(probabilities)?
        } else {
            self.simd_weighting.weights_gains_simd(probabilities)?
        };

        // Calculate prospect value using SIMD if possible
        let prospect_value = if values.len() >= 4 {
            self.calculate_prospect_value_simd(&values, &decision_weights)
        } else {
            values
                .iter()
                .zip(decision_weights.iter())
                .map(|(&value, &weight)| value * weight)
                .sum()
        };

        Ok(prospect_value)
    }

    /// SIMD calculation of final prospect value
    #[cfg(feature = "simd")]
    fn calculate_prospect_value_simd(&self, values: &[f64], weights: &[f64]) -> f64 {
        let mut sum = f64x4::splat(0.0);
        let chunk_size = 4;
        let mut i = 0;

        while i + chunk_size <= values.len() {
            let values_chunk = f64x4::new([
                values[i], values[i + 1], values[i + 2], values[i + 3]
            ]);
            let weights_chunk = f64x4::new([
                weights[i], weights[i + 1], weights[i + 2], weights[i + 3]
            ]);
            
            sum += values_chunk * weights_chunk;
            i += chunk_size;
        }

        // Sum the SIMD register and add remaining elements
        let sum_array = sum.to_array();
        let mut result = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
        
        for j in i..values.len() {
            result += values[j] * weights[j];
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value_function::ValueFunctionParams;
    use crate::probability_weighting::WeightingParams;
    use approx::assert_relative_eq;

    #[test]
    fn test_simd_value_function() {
        let params = ValueFunctionParams::default();
        let simd_vf = SIMDValueFunction::new(params).unwrap();
        
        let outcomes = vec![100.0, 0.0, -100.0, 50.0, -50.0];
        let values = simd_vf.values_simd(&outcomes).unwrap();
        
        assert_eq!(values.len(), outcomes.len());
        assert!(values[0] > 0.0); // Gain
        assert_relative_eq!(values[1], 0.0, epsilon = 1e-10); // Reference point
        assert!(values[2] < 0.0); // Loss
    }

    #[test]
    fn test_simd_probability_weighting() {
        let params = WeightingParams::default();
        let simd_pw = SIMDProbabilityWeighting::new(params).unwrap();
        
        let probabilities = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let weights = simd_pw.weights_gains_simd(&probabilities).unwrap();
        
        assert_eq!(weights.len(), probabilities.len());
        assert!(weights.iter().all(|&w| w >= 0.0 && w <= 1.0));
    }

    #[test]
    fn test_cpu_capabilities() {
        let caps = CPUCapabilities::detect();
        // Test just runs without error - actual capabilities depend on hardware
        println!("AVX2: {}, AVX512: {}, FMA: {}", caps.has_avx2, caps.has_avx512, caps.has_fma);
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::new(4, 100);
        let vec1 = pool.get_value_vec(50);
        let vec2 = pool.get_value_vec(75);
        
        assert!(vec1.capacity() >= 50);
        assert!(vec2.capacity() >= 75);
        
        pool.return_value_vec(vec1);
        pool.return_value_vec(vec2);
    }

    #[test]
    fn test_high_performance_calculator() {
        let value_params = ValueFunctionParams::default();
        let weighting_params = WeightingParams::default();
        let mut calc = HighPerformanceProspectCalculator::new(value_params, weighting_params).unwrap();
        
        let outcomes = vec![100.0, 0.0, -100.0];
        let probabilities = vec![0.3, 0.4, 0.3];
        
        let prospect_value = calc.calculate_prospect_value(&outcomes, &probabilities).unwrap();
        assert!(prospect_value.is_finite());
    }
}