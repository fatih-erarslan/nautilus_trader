//! Parallel processing utilities for LMSR operations

use crate::core::LMSR;
use crate::errors::{LMSRError, Result};
use crate::aggregation::{AggregationMethod, AggregationResult};
use rayon::prelude::*;
use std::sync::Arc;

/// Parallel batch processor for LMSR operations
#[derive(Debug, Clone)]
pub struct ParallelProcessor {
    /// Number of threads to use (0 = auto)
    pub num_threads: usize,
    /// Chunk size for parallel processing
    pub chunk_size: usize,
    /// Enable dynamic load balancing
    pub dynamic_balancing: bool,
}

impl ParallelProcessor {
    /// Create a new parallel processor with default settings
    pub fn new() -> Self {
        Self {
            num_threads: 0, // Use rayon's default (usually num_cpus)
            chunk_size: 1000,
            dynamic_balancing: true,
        }
    }

    /// Create with custom settings
    pub fn with_config(num_threads: usize, chunk_size: usize, dynamic_balancing: bool) -> Self {
        Self {
            num_threads,
            chunk_size,
            dynamic_balancing,
        }
    }

    /// Initialize the thread pool
    pub fn init(&self) -> Result<()> {
        if self.num_threads > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(self.num_threads)
                .build_global()
                .map_err(|e| LMSRError::parallel_error(format!("Failed to initialize thread pool: {}", e)))?;
        }
        Ok(())
    }

    /// Process multiple quantity batches in parallel
    pub fn batch_market_probabilities(
        &self,
        lmsr: &LMSR,
        quantities_batch: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>> {
        if quantities_batch.is_empty() {
            return Ok(Vec::new());
        }

        let lmsr_arc = Arc::new(lmsr.clone());
        
        let results: Result<Vec<Vec<f64>>> = if self.dynamic_balancing {
            // Dynamic load balancing
            quantities_batch
                .par_iter()
                .map(|quantities| {
                    let lmsr_ref = Arc::clone(&lmsr_arc);
                    lmsr_ref.market_probabilities(quantities)
                })
                .collect()
        } else {
            // Static chunking
            quantities_batch
                .par_chunks(self.chunk_size)
                .map(|chunk| {
                    let lmsr_ref = Arc::clone(&lmsr_arc);
                    let mut chunk_results = Vec::with_capacity(chunk.len());
                    for quantities in chunk {
                        chunk_results.push(lmsr_ref.market_probabilities(quantities)?);
                    }
                    Ok(chunk_results)
                })
                .collect::<Result<Vec<Vec<Vec<f64>>>>>()
                .map(|chunks| chunks.into_iter().flatten().collect())
        };

        results
    }

    /// Process multiple cost function calculations in parallel
    pub fn batch_cost_function(
        &self,
        lmsr: &LMSR,
        quantities_batch: &[Vec<f64>],
    ) -> Result<Vec<f64>> {
        if quantities_batch.is_empty() {
            return Ok(Vec::new());
        }

        let lmsr_arc = Arc::new(lmsr.clone());
        
        let results: Result<Vec<f64>> = if self.dynamic_balancing {
            quantities_batch
                .par_iter()
                .map(|quantities| {
                    let lmsr_ref = Arc::clone(&lmsr_arc);
                    lmsr_ref.cost_function(quantities)
                })
                .collect()
        } else {
            quantities_batch
                .par_chunks(self.chunk_size)
                .map(|chunk| {
                    let lmsr_ref = Arc::clone(&lmsr_arc);
                    let mut chunk_results = Vec::with_capacity(chunk.len());
                    for quantities in chunk {
                        chunk_results.push(lmsr_ref.cost_function(quantities)?);
                    }
                    Ok(chunk_results)
                })
                .collect::<Result<Vec<Vec<f64>>>>()
                .map(|chunks| chunks.into_iter().flatten().collect())
        };

        results
    }

    /// Process multiple cost-to-move calculations in parallel
    pub fn batch_cost_to_move(
        &self,
        lmsr: &LMSR,
        current_quantities: &[Vec<f64>],
        target_quantities: &[Vec<f64>],
    ) -> Result<Vec<f64>> {
        if current_quantities.len() != target_quantities.len() {
            return Err(LMSRError::dimension_mismatch(
                current_quantities.len(),
                target_quantities.len(),
            ));
        }

        if current_quantities.is_empty() {
            return Ok(Vec::new());
        }

        let lmsr_arc = Arc::new(lmsr.clone());
        
        let results: Result<Vec<f64>> = if self.dynamic_balancing {
            current_quantities
                .par_iter()
                .zip(target_quantities.par_iter())
                .map(|(current, target)| {
                    let lmsr_ref = Arc::clone(&lmsr_arc);
                    lmsr_ref.cost_to_move(current, target)
                })
                .collect()
        } else {
            current_quantities
                .par_chunks(self.chunk_size)
                .zip(target_quantities.par_chunks(self.chunk_size))
                .map(|(current_chunk, target_chunk)| {
                    let lmsr_ref = Arc::clone(&lmsr_arc);
                    let mut chunk_results = Vec::with_capacity(current_chunk.len());
                    for (current, target) in current_chunk.iter().zip(target_chunk.iter()) {
                        chunk_results.push(lmsr_ref.cost_to_move(current, target)?);
                    }
                    Ok(chunk_results)
                })
                .collect::<Result<Vec<Vec<f64>>>>()
                .map(|chunks| chunks.into_iter().flatten().collect())
        };

        results
    }

    /// Process multiple KL divergence calculations in parallel
    pub fn batch_kl_divergence(
        &self,
        lmsr: &LMSR,
        p_distributions: &[Vec<f64>],
        q_distributions: &[Vec<f64>],
    ) -> Result<Vec<f64>> {
        if p_distributions.len() != q_distributions.len() {
            return Err(LMSRError::dimension_mismatch(
                p_distributions.len(),
                q_distributions.len(),
            ));
        }

        if p_distributions.is_empty() {
            return Ok(Vec::new());
        }

        let lmsr_arc = Arc::new(lmsr.clone());
        
        let results: Result<Vec<f64>> = if self.dynamic_balancing {
            p_distributions
                .par_iter()
                .zip(q_distributions.par_iter())
                .map(|(p, q)| {
                    let lmsr_ref = Arc::clone(&lmsr_arc);
                    lmsr_ref.kl_divergence(p, q)
                })
                .collect()
        } else {
            p_distributions
                .par_chunks(self.chunk_size)
                .zip(q_distributions.par_chunks(self.chunk_size))
                .map(|(p_chunk, q_chunk)| {
                    let lmsr_ref = Arc::clone(&lmsr_arc);
                    let mut chunk_results = Vec::with_capacity(p_chunk.len());
                    for (p, q) in p_chunk.iter().zip(q_chunk.iter()) {
                        chunk_results.push(lmsr_ref.kl_divergence(p, q)?);
                    }
                    Ok(chunk_results)
                })
                .collect::<Result<Vec<Vec<f64>>>>()
                .map(|chunks| chunks.into_iter().flatten().collect())
        };

        results
    }

    /// Process multiple aggregation operations in parallel
    pub fn batch_aggregate_probabilities(
        &self,
        lmsr: &LMSR,
        distribution_batches: &[Vec<Vec<f64>>],
        method: AggregationMethod,
    ) -> Result<Vec<AggregationResult>> {
        if distribution_batches.is_empty() {
            return Ok(Vec::new());
        }

        let lmsr_arc = Arc::new(lmsr.clone());
        
        let results: Result<Vec<AggregationResult>> = if self.dynamic_balancing {
            distribution_batches
                .par_iter()
                .map(|distributions| {
                    let lmsr_ref = Arc::clone(&lmsr_arc);
                    lmsr_ref.aggregate_probabilities(distributions, method)
                })
                .collect()
        } else {
            distribution_batches
                .par_chunks(self.chunk_size)
                .map(|chunk| {
                    let lmsr_ref = Arc::clone(&lmsr_arc);
                    let mut chunk_results = Vec::with_capacity(chunk.len());
                    for distributions in chunk {
                        chunk_results.push(lmsr_ref.aggregate_probabilities(distributions, method)?);
                    }
                    Ok(chunk_results)
                })
                .collect::<Result<Vec<Vec<AggregationResult>>>>()
                .map(|chunks| chunks.into_iter().flatten().collect())
        };

        results
    }

    /// Parallel Monte Carlo simulation for LMSR
    pub fn monte_carlo_simulation(
        &self,
        lmsr: &LMSR,
        initial_quantities: &[f64],
        n_simulations: usize,
        n_steps: usize,
        step_size: f64,
    ) -> Result<Vec<Vec<Vec<f64>>>> {
        if n_simulations == 0 || n_steps == 0 {
            return Ok(Vec::new());
        }

        let lmsr_arc = Arc::new(lmsr.clone());
        let initial_quantities_arc = Arc::new(initial_quantities.to_vec());
        
        let results: Result<Vec<Vec<Vec<f64>>>> = (0..n_simulations)
            .into_par_iter()
            .map(|_| {
                let lmsr_ref = Arc::clone(&lmsr_arc);
                let initial_ref = Arc::clone(&initial_quantities_arc);
                self.single_monte_carlo_path(&lmsr_ref, &initial_ref, n_steps, step_size)
            })
            .collect();

        results
    }

    /// Single Monte Carlo path
    fn single_monte_carlo_path(
        &self,
        lmsr: &LMSR,
        initial_quantities: &[f64],
        n_steps: usize,
        step_size: f64,
    ) -> Result<Vec<Vec<f64>>> {
        use rand::Rng;
        use rand_distr::{Normal, Distribution};
        
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, step_size).map_err(|e| {
            LMSRError::parallel_error(format!("Failed to create normal distribution: {}", e))
        })?;
        
        let mut path = Vec::with_capacity(n_steps + 1);
        let mut current_quantities = initial_quantities.to_vec();
        
        // Add initial state
        path.push(lmsr.market_probabilities(&current_quantities)?);
        
        // Generate path
        for _ in 0..n_steps {
            // Add random shocks to quantities
            for q in &mut current_quantities {
                *q += normal.sample(&mut rng);
            }
            
            // Calculate probabilities
            let probabilities = lmsr.market_probabilities(&current_quantities)?;
            path.push(probabilities);
        }
        
        Ok(path)
    }

    /// Parallel sensitivity analysis
    pub fn sensitivity_analysis(
        &self,
        lmsr: &LMSR,
        base_quantities: &[f64],
        perturbations: &[f64],
    ) -> Result<Vec<Vec<f64>>> {
        if perturbations.is_empty() {
            return Ok(Vec::new());
        }

        let lmsr_arc = Arc::new(lmsr.clone());
        let base_quantities_arc = Arc::new(base_quantities.to_vec());
        
        let results: Result<Vec<Vec<f64>>> = perturbations
            .par_iter()
            .map(|&perturbation| {
                let lmsr_ref = Arc::clone(&lmsr_arc);
                let base_ref = Arc::clone(&base_quantities_arc);
                
                let mut perturbed_quantities = base_ref.clone();
                
                // Apply perturbation to all quantities
                for q in &mut perturbed_quantities {
                    *q += perturbation;
                }
                
                lmsr_ref.market_probabilities(&perturbed_quantities)
            })
            .collect();

        results
    }

    /// Parallel gradient computation (numerical)
    pub fn numerical_gradient(
        &self,
        lmsr: &LMSR,
        quantities: &[f64],
        epsilon: f64,
    ) -> Result<Vec<f64>> {
        if quantities.is_empty() {
            return Ok(Vec::new());
        }

        let lmsr_arc = Arc::new(lmsr.clone());
        let quantities_arc = Arc::new(quantities.to_vec());
        
        let gradients: Result<Vec<f64>> = (0..quantities.len())
            .into_par_iter()
            .map(|i| {
                let lmsr_ref = Arc::clone(&lmsr_arc);
                let quantities_ref = Arc::clone(&quantities_arc);
                
                // Forward difference
                let mut forward_quantities = quantities_ref.clone();
                forward_quantities[i] += epsilon;
                let forward_cost = lmsr_ref.cost_function(&forward_quantities)?;
                
                // Backward difference
                let mut backward_quantities = quantities_ref.clone();
                backward_quantities[i] -= epsilon;
                let backward_cost = lmsr_ref.cost_function(&backward_quantities)?;
                
                // Central difference
                let gradient = (forward_cost - backward_cost) / (2.0 * epsilon);
                Ok(gradient)
            })
            .collect();

        gradients
    }

    /// Parallel Hessian computation (numerical)
    pub fn numerical_hessian(
        &self,
        lmsr: &LMSR,
        quantities: &[f64],
        epsilon: f64,
    ) -> Result<Vec<Vec<f64>>> {
        if quantities.is_empty() {
            return Ok(Vec::new());
        }

        let n = quantities.len();
        let lmsr_arc = Arc::new(lmsr.clone());
        let quantities_arc = Arc::new(quantities.to_vec());
        
        let hessian_rows: Result<Vec<Vec<f64>>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let lmsr_ref = Arc::clone(&lmsr_arc);
                let quantities_ref = Arc::clone(&quantities_arc);
                
                let mut row = Vec::with_capacity(n);
                
                for j in 0..n {
                    if i == j {
                        // Diagonal element: second derivative
                        let mut forward = quantities_ref.clone();
                        forward[i] += epsilon;
                        let forward_cost = lmsr_ref.cost_function(&forward)?;
                        
                        let mut backward = quantities_ref.clone();
                        backward[i] -= epsilon;
                        let backward_cost = lmsr_ref.cost_function(&backward)?;
                        
                        let base_cost = lmsr_ref.cost_function(&quantities_ref)?;
                        let second_derivative = (forward_cost - 2.0 * base_cost + backward_cost) / (epsilon * epsilon);
                        
                        row.push(second_derivative);
                    } else {
                        // Off-diagonal element: mixed partial derivative
                        let mut forward_forward = quantities_ref.clone();
                        forward_forward[i] += epsilon;
                        forward_forward[j] += epsilon;
                        let ff_cost = lmsr_ref.cost_function(&forward_forward)?;
                        
                        let mut forward_backward = quantities_ref.clone();
                        forward_backward[i] += epsilon;
                        forward_backward[j] -= epsilon;
                        let fb_cost = lmsr_ref.cost_function(&forward_backward)?;
                        
                        let mut backward_forward = quantities_ref.clone();
                        backward_forward[i] -= epsilon;
                        backward_forward[j] += epsilon;
                        let bf_cost = lmsr_ref.cost_function(&backward_forward)?;
                        
                        let mut backward_backward = quantities_ref.clone();
                        backward_backward[i] -= epsilon;
                        backward_backward[j] -= epsilon;
                        let bb_cost = lmsr_ref.cost_function(&backward_backward)?;
                        
                        let mixed_derivative = (ff_cost - fb_cost - bf_cost + bb_cost) / (4.0 * epsilon * epsilon);
                        row.push(mixed_derivative);
                    }
                }
                
                Ok(row)
            })
            .collect();

        hessian_rows
    }

    /// Get performance statistics
    pub fn performance_stats(&self) -> ParallelStats {
        ParallelStats {
            num_threads: self.num_threads,
            chunk_size: self.chunk_size,
            dynamic_balancing: self.dynamic_balancing,
            available_parallelism: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1),
        }
    }
}

impl Default for ParallelProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance statistics for parallel processing
#[derive(Debug, Clone)]
pub struct ParallelStats {
    pub num_threads: usize,
    pub chunk_size: usize,
    pub dynamic_balancing: bool,
    pub available_parallelism: usize,
}

impl std::fmt::Display for ParallelStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ParallelStats(threads={}, chunk_size={}, dynamic={}, available={})",
            self.num_threads, self.chunk_size, self.dynamic_balancing, self.available_parallelism
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::LMSR;
    use approx::assert_relative_eq;

    #[test]
    fn test_parallel_processor_creation() {
        let processor = ParallelProcessor::new();
        assert_eq!(processor.num_threads, 0);
        assert_eq!(processor.chunk_size, 1000);
        assert!(processor.dynamic_balancing);
    }

    #[test]
    fn test_batch_market_probabilities() {
        let processor = ParallelProcessor::new();
        let lmsr = LMSR::new(100.0);
        
        let quantities_batch = vec![
            vec![0.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0],
            vec![0.0, 10.0, 0.0],
            vec![0.0, 0.0, 10.0],
        ];
        
        let results = processor.batch_market_probabilities(&lmsr, &quantities_batch).unwrap();
        assert_eq!(results.len(), 4);
        
        // All should sum to 1.0
        for probabilities in &results {
            let sum: f64 = probabilities.iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_batch_cost_function() {
        let processor = ParallelProcessor::new();
        let lmsr = LMSR::new(100.0);
        
        let quantities_batch = vec![
            vec![0.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0],
            vec![0.0, 10.0, 0.0],
        ];
        
        let results = processor.batch_cost_function(&lmsr, &quantities_batch).unwrap();
        assert_eq!(results.len(), 3);
        
        // All costs should be finite
        for cost in &results {
            assert!(cost.is_finite());
        }
    }

    #[test]
    fn test_batch_cost_to_move() {
        let processor = ParallelProcessor::new();
        let lmsr = LMSR::new(100.0);
        
        let current_quantities = vec![
            vec![0.0, 0.0, 0.0],
            vec![5.0, 0.0, 0.0],
        ];
        
        let target_quantities = vec![
            vec![10.0, 0.0, 0.0],
            vec![0.0, 10.0, 0.0],
        ];
        
        let results = processor.batch_cost_to_move(&lmsr, &current_quantities, &target_quantities).unwrap();
        assert_eq!(results.len(), 2);
        
        // All costs should be finite
        for cost in &results {
            assert!(cost.is_finite());
        }
    }

    #[test]
    fn test_monte_carlo_simulation() {
        let processor = ParallelProcessor::new();
        let lmsr = LMSR::new(100.0);
        
        let initial_quantities = vec![0.0, 0.0, 0.0];
        let results = processor.monte_carlo_simulation(&lmsr, &initial_quantities, 10, 5, 0.1).unwrap();
        
        assert_eq!(results.len(), 10); // 10 simulations
        
        for path in &results {
            assert_eq!(path.len(), 6); // 5 steps + initial state
            
            // All probabilities should sum to 1.0
            for probabilities in path {
                let sum: f64 = probabilities.iter().sum();
                assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_sensitivity_analysis() {
        let processor = ParallelProcessor::new();
        let lmsr = LMSR::new(100.0);
        
        let base_quantities = vec![0.0, 0.0, 0.0];
        let perturbations = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        
        let results = processor.sensitivity_analysis(&lmsr, &base_quantities, &perturbations).unwrap();
        assert_eq!(results.len(), 5);
        
        // All should sum to 1.0
        for probabilities in &results {
            let sum: f64 = probabilities.iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_numerical_gradient() {
        let processor = ParallelProcessor::new();
        let lmsr = LMSR::new(100.0);
        
        let quantities = vec![0.0, 0.0, 0.0];
        let gradients = processor.numerical_gradient(&lmsr, &quantities, 0.01).unwrap();
        
        assert_eq!(gradients.len(), 3);
        
        // All gradients should be finite
        for gradient in &gradients {
            assert!(gradient.is_finite());
        }
    }

    #[test]
    fn test_performance_stats() {
        let processor = ParallelProcessor::new();
        let stats = processor.performance_stats();
        
        assert_eq!(stats.num_threads, 0);
        assert_eq!(stats.chunk_size, 1000);
        assert!(stats.dynamic_balancing);
        assert!(stats.available_parallelism >= 1);
    }
}