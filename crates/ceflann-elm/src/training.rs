//! Analytical training algorithms for ELM
//! 
//! Ultra-fast single-pass training using Moore-Penrose pseudoinverse,
//! SVD, and regularized least squares solutions.

use nalgebra::{DMatrix, SVD, QR};
use anyhow::{Result, anyhow};
use tracing::{debug, info, warn};
use std::time::Instant;

#[cfg(feature = "cuda")]
use crate::cuda_kernels::CudaTraining;

use crate::{ELMConfig, TrainingAlgorithm};

/// High-performance analytical training engine
pub struct AnalyticalTraining {
    /// Configuration
    config: ELMConfig,
    
    #[cfg(feature = "cuda")]
    /// CUDA training engine
    cuda_engine: Option<CudaTraining>,
}

impl AnalyticalTraining {
    /// Create new analytical training engine
    pub fn new(config: &ELMConfig) -> Result<Self> {
        #[cfg(feature = "cuda")]
        let cuda_engine = if config.use_cuda {
            Some(CudaTraining::new()?)
        } else {
            None
        };
        
        Ok(Self {
            config: config.clone(),
            #[cfg(feature = "cuda")]
            cuda_engine,
        })
    }
    
    /// Train output weights using analytical solution
    pub fn train(&self, inputs: &DMatrix<f64>, targets: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let start_time = Instant::now();
        
        if inputs.nrows() != targets.nrows() {
            return Err(anyhow!("Input and target sample counts must match"));
        }
        
        if inputs.nrows() == 0 {
            return Err(anyhow!("No training samples provided"));
        }
        
        debug!("Training with {} samples, {} features -> {} outputs", 
               inputs.nrows(), inputs.ncols(), targets.ncols());
        
        let weights = match self.config.training_algorithm {
            TrainingAlgorithm::PseudoInverse => {
                self.train_pseudoinverse(inputs, targets)?
            }
            TrainingAlgorithm::Ridge { lambda } => {
                self.train_ridge_regression(inputs, targets, lambda)?
            }
            TrainingAlgorithm::SVD { tolerance } => {
                self.train_svd(inputs, targets, tolerance)?
            }
        };
        
        let training_time = start_time.elapsed();
        info!("Analytical training completed in {}μs", training_time.as_micros());
        
        Ok(weights)
    }
    
    /// Moore-Penrose pseudoinverse solution: W = (H^T H)^(-1) H^T T
    fn train_pseudoinverse(&self, inputs: &DMatrix<f64>, targets: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        debug!("Using Moore-Penrose pseudoinverse training");
        
        #[cfg(feature = "cuda")]
        if self.config.use_cuda && self.cuda_engine.is_some() {
            return self.cuda_engine.as_ref().unwrap()
                .train_pseudoinverse(inputs, targets);
        }
        
        // CPU implementation using nalgebra
        let h_transpose = inputs.transpose();
        let h_t_h = &h_transpose * inputs;
        
        // Check for numerical stability
        let condition_number = self.estimate_condition_number(&h_t_h)?;
        if condition_number > 1e12 {
            warn!("Matrix is ill-conditioned (condition number: {:.2e}). Consider regularization.", condition_number);
        }
        
        // Compute (H^T H)^(-1)
        let h_t_h_inv = match h_t_h.try_inverse() {
            Some(inv) => inv,
            None => {
                warn!("Matrix inversion failed, falling back to SVD");
                return self.train_svd(inputs, targets, 1e-15);
            }
        };
        
        // Final weights: W = (H^T H)^(-1) H^T T
        let h_t_t = &h_transpose * targets;
        let weights = h_t_h_inv * h_t_t;
        
        Ok(weights)
    }
    
    /// Ridge regression solution: W = (H^T H + λI)^(-1) H^T T
    fn train_ridge_regression(&self, inputs: &DMatrix<f64>, targets: &DMatrix<f64>, lambda: f64) -> Result<DMatrix<f64>> {
        debug!("Using Ridge regression training with λ = {}", lambda);
        
        #[cfg(feature = "cuda")]
        if self.config.use_cuda && self.cuda_engine.is_some() {
            return self.cuda_engine.as_ref().unwrap()
                .train_ridge_regression(inputs, targets, lambda);
        }
        
        let h_transpose = inputs.transpose();
        let mut h_t_h = &h_transpose * inputs;
        
        // Add regularization: H^T H + λI
        for i in 0..h_t_h.nrows() {
            h_t_h[(i, i)] += lambda;
        }
        
        // Compute regularized inverse
        let h_t_h_inv = h_t_h.try_inverse()
            .ok_or_else(|| anyhow!("Regularized matrix inversion failed"))?;
        
        // Final weights
        let h_t_t = &h_transpose * targets;
        let weights = h_t_h_inv * h_t_t;
        
        Ok(weights)
    }
    
    /// SVD-based solution with rank truncation
    fn train_svd(&self, inputs: &DMatrix<f64>, targets: &DMatrix<f64>, tolerance: f64) -> Result<DMatrix<f64>> {
        debug!("Using SVD training with tolerance {}", tolerance);
        
        #[cfg(feature = "cuda")]
        if self.config.use_cuda && self.cuda_engine.is_some() {
            return self.cuda_engine.as_ref().unwrap()
                .train_svd(inputs, targets, tolerance);
        }
        
        // Compute SVD: H = U Σ V^T
        let svd = SVD::new(inputs.clone(), true, true);
        
        let u = svd.u.ok_or_else(|| anyhow!("SVD failed to compute U matrix"))?;
        let v_t = svd.v_t.ok_or_else(|| anyhow!("SVD failed to compute V^T matrix"))?;
        let singular_values = svd.singular_values;
        
        // Find effective rank based on tolerance
        let max_singular_value = singular_values.max();
        let threshold = max_singular_value * tolerance;
        let effective_rank = singular_values.iter()
            .take_while(|&&s| s > threshold)
            .count();
        
        debug!("SVD effective rank: {} / {} (threshold: {:.2e})", 
               effective_rank, singular_values.len(), threshold);
        
        // Compute pseudoinverse using truncated SVD
        let mut sigma_inv = DMatrix::zeros(inputs.ncols(), inputs.nrows());
        for i in 0..effective_rank {
            sigma_inv[(i, i)] = 1.0 / singular_values[i];
        }
        
        // Pseudoinverse: H+ = V Σ+ U^T
        let h_pseudoinverse = &v_t.transpose() * &sigma_inv * &u.transpose();
        
        // Final weights: W = H+ T
        let weights = h_pseudoinverse * targets;
        
        Ok(weights)
    }
    
    /// Estimate condition number for numerical stability assessment
    fn estimate_condition_number(&self, matrix: &DMatrix<f64>) -> Result<f64> {
        let svd = SVD::new(matrix.clone(), false, false);
        let singular_values = svd.singular_values;
        
        if singular_values.is_empty() {
            return Ok(1.0);
        }
        
        let max_sv = singular_values.max();
        let min_sv = singular_values.min();
        
        if min_sv > 1e-15 {
            Ok(max_sv / min_sv)
        } else {
            Ok(f64::INFINITY)
        }
    }
    
    /// Online learning using Recursive Least Squares (RLS)
    pub fn update_online(&self, 
                        current_weights: &DMatrix<f64>,
                        new_input: &DMatrix<f64>,
                        new_target: &DMatrix<f64>,
                        forgetting_factor: f64) -> Result<DMatrix<f64>> {
        // Implement RLS update for true online learning
        // P(k) = (P(k-1) - P(k-1)h(k)h^T(k)P(k-1) / (λ + h^T(k)P(k-1)h(k))) / λ
        // w(k) = w(k-1) + P(k)h(k)(t(k) - h^T(k)w(k-1))
        
        warn!("Online RLS update not yet implemented");
        Ok(current_weights.clone())
    }
    
    /// Batch update with new samples (efficient retraining)
    pub fn update_batch(&self,
                       old_inputs: &DMatrix<f64>,
                       old_targets: &DMatrix<f64>,
                       new_inputs: &DMatrix<f64>,
                       new_targets: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        debug!("Batch update: {} old + {} new samples", 
               old_inputs.nrows(), new_inputs.nrows());
        
        // Simple concatenation and retrain
        // In production, could use incremental algorithms
        let combined_inputs = DMatrix::from_fn(
            old_inputs.nrows() + new_inputs.nrows(),
            old_inputs.ncols(),
            |i, j| {
                if i < old_inputs.nrows() {
                    old_inputs[(i, j)]
                } else {
                    new_inputs[(i - old_inputs.nrows(), j)]
                }
            }
        );
        
        let combined_targets = DMatrix::from_fn(
            old_targets.nrows() + new_targets.nrows(),
            old_targets.ncols(),
            |i, j| {
                if i < old_targets.nrows() {
                    old_targets[(i, j)]
                } else {
                    new_targets[(i - old_targets.nrows(), j)]
                }
            }
        );
        
        self.train(&combined_inputs, &combined_targets)
    }
    
    /// Cross-validation training with multiple folds
    pub fn train_cross_validated(&self,
                                inputs: &DMatrix<f64>,
                                targets: &DMatrix<f64>,
                                n_folds: usize) -> Result<(DMatrix<f64>, f64)> {
        if n_folds < 2 {
            return Err(anyhow!("Number of folds must be at least 2"));
        }
        
        let n_samples = inputs.nrows();
        let fold_size = n_samples / n_folds;
        let mut validation_errors = Vec::new();
        let mut fold_weights = Vec::new();
        
        for fold in 0..n_folds {
            let val_start = fold * fold_size;
            let val_end = if fold == n_folds - 1 { n_samples } else { (fold + 1) * fold_size };
            
            // Create training set (all except validation fold)
            let mut train_inputs_vec = Vec::new();
            let mut train_targets_vec = Vec::new();
            
            for i in 0..n_samples {
                if i < val_start || i >= val_end {
                    train_inputs_vec.push(inputs.row(i).transpose());
                    train_targets_vec.push(targets.row(i).transpose());
                }
            }
            
            // Convert to matrices
            let train_inputs = DMatrix::from_columns(&train_inputs_vec);
            let train_targets = DMatrix::from_columns(&train_targets_vec);
            
            // Train on fold
            let weights = self.train(&train_inputs.transpose(), &train_targets.transpose())?;
            
            // Validate
            let val_inputs = inputs.rows(val_start, val_end - val_start);
            let val_targets = targets.rows(val_start, val_end - val_start);
            let val_predictions = &val_inputs * &weights;
            
            // Calculate validation error (MSE)
            let error = (&val_predictions - &val_targets).norm_squared() / (val_end - val_start) as f64;
            validation_errors.push(error);
            fold_weights.push(weights);
        }
        
        // Find best fold
        let best_fold = validation_errors
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        
        let avg_error = validation_errors.iter().sum::<f64>() / n_folds as f64;
        
        debug!("Cross-validation completed: best fold = {}, avg error = {:.6}", best_fold, avg_error);
        
        Ok((fold_weights[best_fold].clone(), avg_error))
    }
}

/// Training statistics and diagnostics
#[derive(Debug, Clone)]
pub struct TrainingDiagnostics {
    pub training_time_us: u64,
    pub condition_number: f64,
    pub effective_rank: usize,
    pub regularization_used: bool,
    pub training_error: f64,
    pub memory_usage_mb: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use crate::ELMConfig;
    
    #[test]
    fn test_pseudoinverse_training() {
        let config = ELMConfig {
            training_algorithm: TrainingAlgorithm::PseudoInverse,
            use_cuda: false,
            ..Default::default()
        };
        
        let trainer = AnalyticalTraining::new(&config).unwrap();
        
        // Simple linear problem: y = 2x + 1
        let inputs = DMatrix::from_column_slice(3, 2, &[
            1.0, 1.0,  // bias, x
            1.0, 2.0,
            1.0, 3.0,
        ]);
        
        let targets = DMatrix::from_column_slice(3, 1, &[3.0, 5.0, 7.0]);
        
        let weights = trainer.train(&inputs, &targets).unwrap();
        
        // Should recover [1, 2] approximately
        assert_relative_eq!(weights[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(weights[(1, 0)], 2.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_ridge_regression() {
        let config = ELMConfig {
            training_algorithm: TrainingAlgorithm::Ridge { lambda: 0.01 },
            use_cuda: false,
            ..Default::default()
        };
        
        let trainer = AnalyticalTraining::new(&config).unwrap();
        
        // Overdetermined system
        let inputs = DMatrix::from_column_slice(4, 2, &[
            1.0, 1.0,
            1.0, 2.0,
            1.0, 3.0,
            1.0, 4.0,
        ]);
        
        let targets = DMatrix::from_column_slice(4, 1, &[2.1, 3.9, 6.1, 8.0]);
        
        let weights = trainer.train(&inputs, &targets).unwrap();
        
        // Verify reasonable solution
        assert!(weights.nrows() == 2);
        assert!(weights.ncols() == 1);
        assert!(weights[(1, 0)] > 1.5 && weights[(1, 0)] < 2.5); // Should be close to 2
    }
    
    #[test]
    fn test_svd_training() {
        let config = ELMConfig {
            training_algorithm: TrainingAlgorithm::SVD { tolerance: 1e-12 },
            use_cuda: false,
            ..Default::default()
        };
        
        let trainer = AnalyticalTraining::new(&config).unwrap();
        
        // Rank-deficient system
        let inputs = DMatrix::from_column_slice(3, 3, &[
            1.0, 2.0, 3.0,
            2.0, 4.0, 6.0,  // This row is 2x the first row
            0.0, 1.0, 1.0,
        ]);
        
        let targets = DMatrix::from_column_slice(3, 1, &[1.0, 2.0, 0.5]);
        
        let weights = trainer.train(&inputs, &targets).unwrap();
        
        // Should handle rank deficiency gracefully
        assert!(weights.nrows() == 3);
        assert!(weights.ncols() == 1);
    }
    
    #[test]
    fn test_condition_number_estimation() {
        let config = ELMConfig::default();
        let trainer = AnalyticalTraining::new(&config).unwrap();
        
        // Well-conditioned matrix
        let well_conditioned = DMatrix::identity(3, 3);
        let cond_num = trainer.estimate_condition_number(&well_conditioned).unwrap();
        assert_relative_eq!(cond_num, 1.0, epsilon = 1e-10);
        
        // Ill-conditioned matrix
        let mut ill_conditioned = DMatrix::identity(3, 3);
        ill_conditioned[(2, 2)] = 1e-15;
        let cond_num = trainer.estimate_condition_number(&ill_conditioned).unwrap();
        assert!(cond_num > 1e10);
    }
}