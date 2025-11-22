//! Simplified high-performance optimizations for prospect theory calculations
//! 
//! This module provides basic optimizations that don't require complex dependencies.

use crate::errors::{ProspectTheoryError, Result};
use crate::value_function::{ValueFunction, ValueFunctionParams};
use crate::probability_weighting::{ProbabilityWeighting, WeightingParams};

use std::sync::Arc;

/// Simple parallel batch processor using standard library threading
pub struct SimpleBatchProcessor {
    value_function: Arc<ValueFunction>,
    probability_weighting: Arc<ProbabilityWeighting>,
    thread_count: usize,
}

impl SimpleBatchProcessor {
    /// Create new simple batch processor
    pub fn new(
        value_params: ValueFunctionParams,
        weighting_params: WeightingParams,
        thread_count: Option<usize>,
    ) -> Result<Self> {
        let value_function = Arc::new(ValueFunction::new(value_params)?);
        let probability_weighting = Arc::new(ProbabilityWeighting::new(weighting_params, 
            crate::probability_weighting::WeightingFunction::TverskyKahneman)?);

        Ok(Self {
            value_function,
            probability_weighting,
            thread_count: thread_count.unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)),
        })
    }

    /// Process batch of prospect calculations
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

        // Simple sequential processing for small batches
        if outcomes_batch.len() < 100 {
            return outcomes_batch
                .iter()
                .zip(probabilities_batch.iter())
                .map(|(outcomes, probabilities)| {
                    self.calculate_single_prospect(outcomes, probabilities)
                })
                .collect();
        }

        // Use threading for larger batches
        self.process_parallel(outcomes_batch, probabilities_batch)
    }

    /// Process using standard library threading
    fn process_parallel(
        &self,
        outcomes_batch: &[Vec<f64>],
        probabilities_batch: &[Vec<f64>],
    ) -> Result<Vec<f64>> {
        let chunk_size = (outcomes_batch.len() / self.thread_count).max(1);
        let mut handles = Vec::new();
        let mut results = vec![0.0; outcomes_batch.len()];
        
        // Process in chunks using threads
        for (chunk_idx, (outcome_chunk, prob_chunk)) in outcomes_batch.chunks(chunk_size)
            .zip(probabilities_batch.chunks(chunk_size))
            .enumerate() {
            
            let vf = Arc::clone(&self.value_function);
            let pw = Arc::clone(&self.probability_weighting);
            let outcomes = outcome_chunk.to_vec();
            let probabilities = prob_chunk.to_vec();
            
            let handle = std::thread::spawn(move || {
                let mut chunk_results = Vec::new();
                for (outcomes, probabilities) in outcomes.iter().zip(probabilities.iter()) {
                    match Self::calculate_prospect_static(&vf, &pw, outcomes, probabilities) {
                        Ok(result) => chunk_results.push(result),
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
                ProspectTheoryError::computation_failed("Thread join failed")
            )??;
            
            for (i, result) in chunk_results.into_iter().enumerate() {
                if start_idx + i < results.len() {
                    results[start_idx + i] = result;
                }
            }
        }
        
        Ok(results)
    }

    /// Calculate single prospect value
    fn calculate_single_prospect(&self, outcomes: &[f64], probabilities: &[f64]) -> Result<f64> {
        Self::calculate_prospect_static(&self.value_function, &self.probability_weighting, outcomes, probabilities)
    }

    /// Static method for thread-safe calculation
    fn calculate_prospect_static(
        vf: &ValueFunction,
        pw: &ProbabilityWeighting,
        outcomes: &[f64],
        probabilities: &[f64],
    ) -> Result<f64> {
        let values = vf.values(outcomes)?;
        let decision_weights = pw.decision_weights(probabilities, outcomes)?;
        
        let prospect_value: f64 = values
            .iter()
            .zip(decision_weights.iter())
            .map(|(&value, &weight)| value * weight)
            .sum();

        Ok(prospect_value)
    }
}

/// Simple cache for common calculations
pub struct SimpleCache {
    power_cache: std::sync::RwLock<std::collections::HashMap<(u64, u64), f64>>,
    max_size: usize,
}

impl SimpleCache {
    /// Create new simple cache
    pub fn new(max_size: usize) -> Self {
        Self {
            power_cache: std::sync::RwLock::new(std::collections::HashMap::new()),
            max_size,
        }
    }

    /// Cached power calculation
    pub fn cached_pow(&self, base: f64, exponent: f64) -> f64 {
        let base_bits = base.to_bits();
        let exp_bits = exponent.to_bits();
        let key = (base_bits, exp_bits);

        // Try cache first
        {
            let cache = self.power_cache.read().unwrap();
            if let Some(&result) = cache.get(&key) {
                return result;
            }
        }

        // Compute and cache
        let result = base.powf(exponent);
        {
            let mut cache = self.power_cache.write().unwrap();
            if cache.len() < self.max_size {
                cache.insert(key, result);
            }
        }
        
        result
    }

    /// Clear cache
    pub fn clear(&self) {
        let mut cache = self.power_cache.write().unwrap();
        cache.clear();
    }

    /// Get cache size
    pub fn size(&self) -> usize {
        let cache = self.power_cache.read().unwrap();
        cache.len()
    }
}

/// Performance measurement utilities
pub struct PerformanceMeasurer {
    measurements: std::sync::RwLock<Vec<std::time::Duration>>,
}

impl PerformanceMeasurer {
    /// Create new performance measurer
    pub fn new() -> Self {
        Self {
            measurements: std::sync::RwLock::new(Vec::new()),
        }
    }

    /// Measure execution time of a function
    pub fn measure<F, R>(&self, func: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = std::time::Instant::now();
        let result = func();
        let duration = start.elapsed();
        
        let mut measurements = self.measurements.write().unwrap();
        measurements.push(duration);
        
        // Keep only recent measurements
        if measurements.len() > 1000 {
            measurements.drain(0..500);
        }
        
        result
    }

    /// Get average execution time
    pub fn average_time(&self) -> std::time::Duration {
        let measurements = self.measurements.read().unwrap();
        if measurements.is_empty() {
            return std::time::Duration::new(0, 0);
        }
        
        let total: std::time::Duration = measurements.iter().sum();
        total / measurements.len() as u32
    }

    /// Get percentile execution time
    pub fn percentile_time(&self, percentile: f64) -> std::time::Duration {
        let measurements = self.measurements.read().unwrap();
        if measurements.is_empty() {
            return std::time::Duration::new(0, 0);
        }
        
        let mut sorted = measurements.clone();
        sorted.sort();
        
        let index = ((sorted.len() as f64 - 1.0) * percentile / 100.0) as usize;
        sorted[index.min(sorted.len() - 1)]
    }

    /// Get measurement count
    pub fn count(&self) -> usize {
        let measurements = self.measurements.read().unwrap();
        measurements.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value_function::ValueFunctionParams;
    use crate::probability_weighting::WeightingParams;

    #[test]
    fn test_simple_batch_processor() {
        let value_params = ValueFunctionParams::default();
        let weighting_params = WeightingParams::default();
        let processor = SimpleBatchProcessor::new(value_params, weighting_params, Some(2)).unwrap();
        
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
        assert!(results.iter().all(|&r| r.is_finite()));
    }

    #[test]
    fn test_simple_cache() {
        let cache = SimpleCache::new(100);
        
        let result1 = cache.cached_pow(2.0, 3.0);
        let result2 = cache.cached_pow(2.0, 3.0); // Should hit cache
        
        assert_eq!(result1, result2);
        assert_eq!(result1, 8.0);
        assert!(cache.size() > 0);
    }

    #[test]
    fn test_performance_measurer() {
        let measurer = PerformanceMeasurer::new();
        
        let result = measurer.measure(|| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            42
        });
        
        assert_eq!(result, 42);
        assert!(measurer.average_time() >= std::time::Duration::from_millis(10));
        assert_eq!(measurer.count(), 1);
    }
}