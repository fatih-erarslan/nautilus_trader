//! ML Utilities and Helper Functions
//!
//! This module provides various utility functions and helpers for ML operations,
//! including data preprocessing, feature engineering, and evaluation metrics.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use std::collections::HashMap;
use crate::ml::{MLError, MLResult, PerformanceMetrics};

/// Feature importance calculation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureImportanceMethod {
    /// Permutation importance
    Permutation,
    /// Correlation with target
    Correlation,
    /// Mutual information
    MutualInformation,
    /// SHAP values (simplified)
    SHAP,
    /// Variance-based importance
    Variance,
}

/// Data splitting strategies
#[derive(Debug, Clone)]
pub enum DataSplit {
    /// Random split
    Random { train_ratio: f64, seed: Option<u64> },
    /// Time-based split
    Temporal { split_point: usize },
    /// Stratified split (for classification)
    Stratified { train_ratio: f64, seed: Option<u64> },
}

/// Feature scaling methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingMethod {
    /// Standard scaling (z-score normalization)
    Standard,
    /// Min-Max scaling to [0, 1]
    MinMax,
    /// Robust scaling (median and IQR)
    Robust,
    /// Unit vector scaling
    Unit,
    /// Max absolute scaling
    MaxAbs,
}

/// Evaluation metric types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    /// Classification metrics
    Classification,
    /// Regression metrics
    Regression,
    /// Ranking metrics
    Ranking,
    /// Clustering metrics
    Clustering,
}

/// Data preprocessor utility
pub struct DataPreprocessor {
    /// Scaling method
    scaling_method: ScalingMethod,
    /// Learned parameters for scaling
    scaling_params: Option<ScalingParams>,
    /// Handle missing values
    handle_missing: bool,
    /// Missing value strategy
    missing_strategy: MissingValueStrategy,
}

/// Scaling parameters
#[derive(Debug, Clone)]
pub struct ScalingParams {
    /// Feature means (for standard scaling)
    pub means: Array1<f64>,
    /// Feature standard deviations (for standard scaling)
    pub stds: Array1<f64>,
    /// Feature minimums (for min-max scaling)
    pub mins: Array1<f64>,
    /// Feature maximums (for min-max scaling)
    pub maxs: Array1<f64>,
    /// Feature medians (for robust scaling)
    pub medians: Array1<f64>,
    /// Interquartile ranges (for robust scaling)
    pub iqrs: Array1<f64>,
}

/// Missing value handling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingValueStrategy {
    /// Remove rows with missing values
    Drop,
    /// Fill with mean value
    Mean,
    /// Fill with median value
    Median,
    /// Fill with mode (most frequent value)
    Mode,
    /// Fill with constant value
    Constant(f64),
    /// Forward fill
    ForwardFill,
    /// Backward fill
    BackwardFill,
}

impl DataPreprocessor {
    /// Create new data preprocessor
    pub fn new(scaling_method: ScalingMethod) -> Self {
        Self {
            scaling_method,
            scaling_params: None,
            handle_missing: true,
            missing_strategy: MissingValueStrategy::Mean,
        }
    }
    
    /// Set missing value handling strategy
    pub fn with_missing_strategy(mut self, strategy: MissingValueStrategy) -> Self {
        self.missing_strategy = strategy;
        self
    }
    
    /// Fit preprocessor to training data
    pub fn fit(&mut self, data: &Array2<f64>) -> MLResult<()> {
        let n_features = data.ncols();
        
        match self.scaling_method {
            ScalingMethod::Standard => {
                let means = data.mean_axis(Axis(0)).unwrap();
                let stds = data.std_axis(Axis(0), 0.0);
                
                self.scaling_params = Some(ScalingParams {
                    means,
                    stds,
                    mins: Array1::zeros(n_features),
                    maxs: Array1::zeros(n_features),
                    medians: Array1::zeros(n_features),
                    iqrs: Array1::zeros(n_features),
                });
            }
            ScalingMethod::MinMax => {
                let mins = data.fold_axis(Axis(0), f64::INFINITY, |&acc, &x| acc.min(x));
                let maxs = data.fold_axis(Axis(0), f64::NEG_INFINITY, |&acc, &x| acc.max(x));
                
                self.scaling_params = Some(ScalingParams {
                    means: Array1::zeros(n_features),
                    stds: Array1::zeros(n_features),
                    mins,
                    maxs,
                    medians: Array1::zeros(n_features),
                    iqrs: Array1::zeros(n_features),
                });
            }
            ScalingMethod::Robust => {
                let medians = self.compute_medians(data);
                let iqrs = self.compute_iqrs(data, &medians);
                
                self.scaling_params = Some(ScalingParams {
                    means: Array1::zeros(n_features),
                    stds: Array1::zeros(n_features),
                    mins: Array1::zeros(n_features),
                    maxs: Array1::zeros(n_features),
                    medians,
                    iqrs,
                });
            }
            ScalingMethod::Unit | ScalingMethod::MaxAbs => {
                // These methods don't require fitting
                self.scaling_params = Some(ScalingParams {
                    means: Array1::zeros(n_features),
                    stds: Array1::zeros(n_features),
                    mins: Array1::zeros(n_features),
                    maxs: Array1::zeros(n_features),
                    medians: Array1::zeros(n_features),
                    iqrs: Array1::zeros(n_features),
                });
            }
        }
        
        Ok(())
    }
    
    /// Transform data using fitted parameters
    pub fn transform(&self, data: &Array2<f64>) -> MLResult<Array2<f64>> {
        let params = self.scaling_params.as_ref()
            .ok_or_else(|| MLError::PreprocessingError {
                message: "Preprocessor not fitted".to_string(),
            })?;
        
        let mut transformed = data.clone();
        
        // Handle missing values first
        if self.handle_missing {
            self.handle_missing_values(&mut transformed)?;
        }
        
        // Apply scaling
        match self.scaling_method {
            ScalingMethod::Standard => {
                for mut row in transformed.rows_mut() {
                    for (i, val) in row.iter_mut().enumerate() {
                        if params.stds[i] > 1e-8 {
                            *val = (*val - params.means[i]) / params.stds[i];
                        }
                    }
                }
            }
            ScalingMethod::MinMax => {
                for mut row in transformed.rows_mut() {
                    for (i, val) in row.iter_mut().enumerate() {
                        let range = params.maxs[i] - params.mins[i];
                        if range > 1e-8 {
                            *val = (*val - params.mins[i]) / range;
                        }
                    }
                }
            }
            ScalingMethod::Robust => {
                for mut row in transformed.rows_mut() {
                    for (i, val) in row.iter_mut().enumerate() {
                        if params.iqrs[i] > 1e-8 {
                            *val = (*val - params.medians[i]) / params.iqrs[i];
                        }
                    }
                }
            }
            ScalingMethod::Unit => {
                for mut row in transformed.rows_mut() {
                    let norm = (row.iter().map(|&x| x * x).sum::<f64>()).sqrt();
                    if norm > 1e-8 {
                        for val in row.iter_mut() {
                            *val /= norm;
                        }
                    }
                }
            }
            ScalingMethod::MaxAbs => {
                let max_abs = transformed.fold_axis(Axis(0), 0.0, |&acc, &x| acc.max(x.abs()));
                for mut row in transformed.rows_mut() {
                    for (i, val) in row.iter_mut().enumerate() {
                        if max_abs[i] > 1e-8 {
                            *val /= max_abs[i];
                        }
                    }
                }
            }
        }
        
        Ok(transformed)
    }
    
    /// Fit and transform in one step
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> MLResult<Array2<f64>> {
        self.fit(data)?;
        self.transform(data)
    }
    
    /// Handle missing values
    fn handle_missing_values(&self, data: &mut Array2<f64>) -> MLResult<()> {
        match self.missing_strategy {
            MissingValueStrategy::Mean => {
                let means = data.mean_axis(Axis(0)).unwrap();
                for mut row in data.rows_mut() {
                    for (i, val) in row.iter_mut().enumerate() {
                        if val.is_nan() {
                            *val = means[i];
                        }
                    }
                }
            }
            MissingValueStrategy::Median => {
                let medians = self.compute_medians(data);
                for mut row in data.rows_mut() {
                    for (i, val) in row.iter_mut().enumerate() {
                        if val.is_nan() {
                            *val = medians[i];
                        }
                    }
                }
            }
            MissingValueStrategy::Constant(fill_value) => {
                for val in data.iter_mut() {
                    if val.is_nan() {
                        *val = fill_value;
                    }
                }
            }
            _ => {
                // Other strategies would be implemented here
                return Err(MLError::PreprocessingError {
                    message: "Missing value strategy not implemented".to_string(),
                });
            }
        }
        
        Ok(())
    }
    
    /// Compute medians for each column
    fn compute_medians(&self, data: &Array2<f64>) -> Array1<f64> {
        let n_features = data.ncols();
        let mut medians = Array1::zeros(n_features);
        
        for (i, mut column_median) in medians.iter_mut().enumerate() {
            let mut column_values: Vec<f64> = data.column(i)
                .iter()
                .filter(|&&x| !x.is_nan())
                .cloned()
                .collect();
            
            if !column_values.is_empty() {
                column_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mid = column_values.len() / 2;
                
                *column_median = if column_values.len() % 2 == 0 {
                    (column_values[mid - 1] + column_values[mid]) / 2.0
                } else {
                    column_values[mid]
                };
            }
        }
        
        medians
    }
    
    /// Compute interquartile ranges
    fn compute_iqrs(&self, data: &Array2<f64>, medians: &Array1<f64>) -> Array1<f64> {
        let n_features = data.ncols();
        let mut iqrs = Array1::zeros(n_features);
        
        for (i, mut column_iqr) in iqrs.iter_mut().enumerate() {
            let column_values: Vec<f64> = data.column(i)
                .iter()
                .filter(|&&x| !x.is_nan())
                .map(|&x| (x - medians[i]).abs())
                .collect();
            
            if !column_values.is_empty() {
                let mut sorted_values = column_values.clone();
                sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                
                let q75_idx = (sorted_values.len() as f64 * 0.75) as usize;
                let q25_idx = (sorted_values.len() as f64 * 0.25) as usize;
                
                if q75_idx < sorted_values.len() && q25_idx < sorted_values.len() {
                    *column_iqr = sorted_values[q75_idx] - sorted_values[q25_idx];
                    if *column_iqr < 1e-8 {
                        *column_iqr = 1.0; // Avoid division by zero
                    }
                } else {
                    *column_iqr = 1.0;
                }
            } else {
                *column_iqr = 1.0;
            }
        }
        
        iqrs
    }
}

/// Feature importance calculator
pub struct FeatureImportance;

impl FeatureImportance {
    /// Calculate feature importance using specified method
    pub fn calculate(
        method: FeatureImportanceMethod,
        features: &Array2<f64>,
        targets: &Array1<f64>,
    ) -> MLResult<Array1<f64>> {
        match method {
            FeatureImportanceMethod::Correlation => {
                Self::correlation_importance(features, targets)
            }
            FeatureImportanceMethod::Variance => {
                Self::variance_importance(features)
            }
            FeatureImportanceMethod::Permutation => {
                // Simplified implementation
                Self::correlation_importance(features, targets)
            }
            FeatureImportanceMethod::MutualInformation => {
                // Simplified implementation
                Self::correlation_importance(features, targets)
            }
            FeatureImportanceMethod::SHAP => {
                // Simplified implementation
                Self::correlation_importance(features, targets)
            }
        }
    }
    
    /// Calculate correlation-based importance
    fn correlation_importance(features: &Array2<f64>, targets: &Array1<f64>) -> MLResult<Array1<f64>> {
        let n_features = features.ncols();
        let mut importance = Array1::zeros(n_features);
        
        let target_mean = targets.mean().unwrap_or(0.0);
        
        for (i, mut imp) in importance.iter_mut().enumerate() {
            let feature_col = features.column(i);
            let feature_mean = feature_col.mean().unwrap_or(0.0);
            
            let mut numerator = 0.0;
            let mut sum_sq_feature = 0.0;
            let mut sum_sq_target = 0.0;
            
            for (&feature_val, &target_val) in feature_col.iter().zip(targets.iter()) {
                let feature_diff = feature_val - feature_mean;
                let target_diff = target_val - target_mean;
                
                numerator += feature_diff * target_diff;
                sum_sq_feature += feature_diff * feature_diff;
                sum_sq_target += target_diff * target_diff;
            }
            
            let denominator = (sum_sq_feature * sum_sq_target).sqrt();
            
            if denominator > 1e-8 {
                *imp = (numerator / denominator).abs();
            }
        }
        
        Ok(importance)
    }
    
    /// Calculate variance-based importance
    fn variance_importance(features: &Array2<f64>) -> MLResult<Array1<f64>> {
        let variances = features.var_axis(Axis(0), 0.0);
        
        // Normalize variances to [0, 1] range
        let max_var = variances.fold(0.0, |acc, &x| acc.max(x));
        if max_var > 1e-8 {
            Ok(variances / max_var)
        } else {
            Ok(variances)
        }
    }
}

/// Evaluation metrics calculator
pub struct EvaluationMetrics;

impl EvaluationMetrics {
    /// Calculate comprehensive performance metrics
    pub fn calculate(
        metric_type: MetricType,
        predictions: &Array1<f64>,
        targets: &Array1<f64>,
    ) -> MLResult<PerformanceMetrics> {
        match metric_type {
            MetricType::Classification => Self::classification_metrics(predictions, targets),
            MetricType::Regression => Self::regression_metrics(predictions, targets),
            MetricType::Ranking => Self::ranking_metrics(predictions, targets),
            MetricType::Clustering => Self::clustering_metrics(predictions, targets),
        }
    }
    
    /// Calculate classification metrics
    fn classification_metrics(predictions: &Array1<f64>, targets: &Array1<f64>) -> MLResult<PerformanceMetrics> {
        if predictions.len() != targets.len() {
            return Err(MLError::DimensionMismatch {
                expected: format!("{} predictions", targets.len()),
                actual: format!("{} predictions", predictions.len()),
            });
        }
        
        let n_samples = predictions.len();
        let mut correct = 0;
        let mut true_positives = 0;
        let mut false_positives = 0;
        let mut false_negatives = 0;
        let mut true_negatives = 0;
        
        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            let pred_class = if pred > 0.5 { 1.0 } else { 0.0 };
            let target_class = if target > 0.5 { 1.0 } else { 0.0 };
            
            if (pred_class - target_class).abs() < 0.1 {
                correct += 1;
            }
            
            match (pred_class > 0.5, target_class > 0.5) {
                (true, true) => true_positives += 1,
                (true, false) => false_positives += 1,
                (false, true) => false_negatives += 1,
                (false, false) => true_negatives += 1,
            }
        }
        
        let accuracy = correct as f64 / n_samples as f64;
        
        let precision = if true_positives + false_positives > 0 {
            true_positives as f64 / (true_positives + false_positives) as f64
        } else {
            0.0
        };
        
        let recall = if true_positives + false_negatives > 0 {
            true_positives as f64 / (true_positives + false_negatives) as f64
        } else {
            0.0
        };
        
        let f1_score = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        
        let mut metrics = PerformanceMetrics::default();
        metrics.accuracy = Some(accuracy);
        metrics.precision = Some(precision);
        metrics.recall = Some(recall);
        metrics.f1_score = Some(f1_score);
        
        Ok(metrics)
    }
    
    /// Calculate regression metrics
    fn regression_metrics(predictions: &Array1<f64>, targets: &Array1<f64>) -> MLResult<PerformanceMetrics> {
        if predictions.len() != targets.len() {
            return Err(MLError::DimensionMismatch {
                expected: format!("{} predictions", targets.len()),
                actual: format!("{} predictions", predictions.len()),
            });
        }
        
        let n_samples = predictions.len() as f64;
        let target_mean = targets.mean().unwrap_or(0.0);
        
        let mut sum_squared_error = 0.0;
        let mut sum_absolute_error = 0.0;
        let mut sum_squared_total = 0.0;
        
        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            let error = target - pred;
            sum_squared_error += error * error;
            sum_absolute_error += error.abs();
            sum_squared_total += (target - target_mean) * (target - target_mean);
        }
        
        let mse = sum_squared_error / n_samples;
        let rmse = mse.sqrt();
        let mae = sum_absolute_error / n_samples;
        
        let r2_score = if sum_squared_total > 1e-8 {
            1.0 - (sum_squared_error / sum_squared_total)
        } else {
            0.0
        };
        
        let mut metrics = PerformanceMetrics::default();
        metrics.mse = Some(mse);
        metrics.rmse = Some(rmse);
        metrics.mae = Some(mae);
        metrics.r2_score = Some(r2_score);
        
        Ok(metrics)
    }
    
    /// Calculate ranking metrics
    fn ranking_metrics(predictions: &Array1<f64>, targets: &Array1<f64>) -> MLResult<PerformanceMetrics> {
        // Simplified ranking metrics implementation
        // In practice, this would include metrics like NDCG, MAP, etc.
        Self::regression_metrics(predictions, targets)
    }
    
    /// Calculate clustering metrics
    fn clustering_metrics(predictions: &Array1<f64>, targets: &Array1<f64>) -> MLResult<PerformanceMetrics> {
        // Simplified clustering metrics implementation
        // In practice, this would include metrics like silhouette score, adjusted rand index, etc.
        let mut metrics = PerformanceMetrics::default();
        
        // Calculate adjusted rand index (simplified)
        let ari = Self::adjusted_rand_index(predictions, targets);
        metrics.add_custom("adjusted_rand_index".to_string(), ari);
        
        Ok(metrics)
    }
    
    /// Calculate adjusted rand index (simplified implementation)
    fn adjusted_rand_index(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
        // This is a very simplified implementation
        // A proper implementation would compute the full ARI formula
        let mut agreement = 0;
        let n = predictions.len();
        
        for i in 0..n {
            for j in (i + 1)..n {
                let pred_same = (predictions[i] - predictions[j]).abs() < 0.5;
                let target_same = (targets[i] - targets[j]).abs() < 0.5;
                
                if pred_same == target_same {
                    agreement += 1;
                }
            }
        }
        
        let total_pairs = n * (n - 1) / 2;
        if total_pairs > 0 {
            agreement as f64 / total_pairs as f64
        } else {
            0.0
        }
    }
}

/// Data splitter utility
pub struct DataSplitter;

impl DataSplitter {
    /// Split data according to strategy
    pub fn split(
        strategy: &DataSplit,
        features: &Array2<f64>,
        targets: &Array1<f64>,
    ) -> MLResult<(Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>)> {
        match strategy {
            DataSplit::Random { train_ratio, seed } => {
                Self::random_split(features, targets, *train_ratio, *seed)
            }
            DataSplit::Temporal { split_point } => {
                Self::temporal_split(features, targets, *split_point)
            }
            DataSplit::Stratified { train_ratio, seed } => {
                // For simplicity, use random split
                Self::random_split(features, targets, *train_ratio, *seed)
            }
        }
    }
    
    /// Random split
    fn random_split(
        features: &Array2<f64>,
        targets: &Array1<f64>,
        train_ratio: f64,
        seed: Option<u64>,
    ) -> MLResult<(Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>)> {
        let n_samples = features.nrows();
        let n_train = (n_samples as f64 * train_ratio) as usize;
        
        let mut indices: Vec<usize> = (0..n_samples).collect();
        
        // Shuffle indices if seed is provided
        if let Some(seed_val) = seed {
            use rand::{SeedableRng, seq::SliceRandom};
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed_val);
            indices.shuffle(&mut rng);
        }
        
        // Split indices
        let train_indices = &indices[0..n_train];
        let test_indices = &indices[n_train..];
        
        // Create train set
        let mut x_train = Array2::zeros((n_train, features.ncols()));
        let mut y_train = Array1::zeros(n_train);
        
        for (i, &idx) in train_indices.iter().enumerate() {
            x_train.row_mut(i).assign(&features.row(idx));
            y_train[i] = targets[idx];
        }
        
        // Create test set
        let n_test = test_indices.len();
        let mut x_test = Array2::zeros((n_test, features.ncols()));
        let mut y_test = Array1::zeros(n_test);
        
        for (i, &idx) in test_indices.iter().enumerate() {
            x_test.row_mut(i).assign(&features.row(idx));
            y_test[i] = targets[idx];
        }
        
        Ok((x_train, y_train, x_test, y_test))
    }
    
    /// Temporal split
    fn temporal_split(
        features: &Array2<f64>,
        targets: &Array1<f64>,
        split_point: usize,
    ) -> MLResult<(Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>)> {
        if split_point >= features.nrows() {
            return Err(MLError::ValidationError {
                message: "Split point exceeds data size".to_string(),
            });
        }
        
        let x_train = features.slice(s![0..split_point, ..]).to_owned();
        let y_train = targets.slice(s![0..split_point]).to_owned();
        let x_test = features.slice(s![split_point.., ..]).to_owned();
        let y_test = targets.slice(s![split_point..]).to_owned();
        
        Ok((x_train, y_train, x_test, y_test))
    }
}

/// Dummy feature extractor for testing
pub struct DummyFeatureExtractor;

impl crate::traits::FeatureExtractor for DummyFeatureExtractor {
    fn extract(&self, data: &Array2<f32>) -> crate::ml::MLResult<Array2<f32>> {
        Ok(data.clone())
    }
    
    fn name(&self) -> &str {
        "DummyFeatureExtractor"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;
    use rand_distr::Uniform;
    
    #[test]
    fn test_data_preprocessor_standard_scaling() {
        let data = Array2::from_shape_vec(
            (4, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ).unwrap();
        
        let mut preprocessor = DataPreprocessor::new(ScalingMethod::Standard);
        let transformed = preprocessor.fit_transform(&data).unwrap();
        
        // Check that transformed data has mean ~0 and std ~1
        let transformed_mean = transformed.mean_axis(Axis(0)).unwrap();
        let transformed_std = transformed.std_axis(Axis(0), 0.0);
        
        for &mean_val in transformed_mean.iter() {
            assert_abs_diff_eq!(mean_val, 0.0, epsilon = 1e-10);
        }
        
        for &std_val in transformed_std.iter() {
            assert_abs_diff_eq!(std_val, 1.0, epsilon = 1e-10);
        }
    }
    
    #[test]
    fn test_data_preprocessor_minmax_scaling() {
        let data = Array2::from_shape_vec(
            (3, 2),
            vec![1.0, 2.0, 5.0, 10.0, 3.0, 6.0],
        ).unwrap();
        
        let mut preprocessor = DataPreprocessor::new(ScalingMethod::MinMax);
        let transformed = preprocessor.fit_transform(&data).unwrap();
        
        // Check that all values are in [0, 1] range
        for &val in transformed.iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }
        
        // Check that min and max values are preserved
        let col0_min = transformed.column(0).fold(f64::INFINITY, |acc, &x| acc.min(x));
        let col0_max = transformed.column(0).fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        
        assert_abs_diff_eq!(col0_min, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(col0_max, 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_feature_importance_correlation() {
        let features = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 2.0, 0.5,  // Feature perfectly correlated with target
                2.0, 4.0, 1.0,
                3.0, 6.0, 1.5,
                4.0, 8.0, 2.0,
                5.0, 10.0, 2.5,
            ],
        ).unwrap();
        
        let targets = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        
        let importance = FeatureImportance::calculate(
            FeatureImportanceMethod::Correlation,
            &features,
            &targets,
        ).unwrap();
        
        // First feature should have highest importance (perfect correlation)
        assert!(importance[0] > importance[1]);
        assert!(importance[0] > importance[2]);
        
        // All importance values should be between 0 and 1
        for &imp in importance.iter() {
            assert!(imp >= 0.0 && imp <= 1.0);
        }
    }
    
    #[test]
    fn test_evaluation_metrics_classification() {
        let predictions = Array1::from_vec(vec![0.8, 0.3, 0.9, 0.1, 0.7]);
        let targets = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0]);
        
        let metrics = EvaluationMetrics::calculate(
            MetricType::Classification,
            &predictions,
            &targets,
        ).unwrap();
        
        assert!(metrics.accuracy.is_some());
        assert!(metrics.precision.is_some());
        assert!(metrics.recall.is_some());
        assert!(metrics.f1_score.is_some());
        
        let accuracy = metrics.accuracy.unwrap();
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
        assert_eq!(accuracy, 1.0); // All predictions are correct
    }
    
    #[test]
    fn test_evaluation_metrics_regression() {
        let predictions = Array1::from_vec(vec![1.1, 2.0, 2.9, 4.1, 5.0]);
        let targets = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        
        let metrics = EvaluationMetrics::calculate(
            MetricType::Regression,
            &predictions,
            &targets,
        ).unwrap();
        
        assert!(metrics.mse.is_some());
        assert!(metrics.rmse.is_some());
        assert!(metrics.mae.is_some());
        assert!(metrics.r2_score.is_some());
        
        let mse = metrics.mse.unwrap();
        let r2 = metrics.r2_score.unwrap();
        
        assert!(mse > 0.0);
        assert!(r2 > 0.9); // Should be high R² for this close prediction
    }
    
    #[test]
    fn test_data_splitter_random() {
        let features = Array2::random((100, 5), Uniform::new(-1.0, 1.0));
        let targets = Array1::random(100, Uniform::new(0.0, 1.0));
        
        let split = DataSplit::Random {
            train_ratio: 0.8,
            seed: Some(42),
        };
        
        let (x_train, y_train, x_test, y_test) = DataSplitter::split(&split, &features, &targets).unwrap();
        
        assert_eq!(x_train.nrows(), 80);
        assert_eq!(y_train.len(), 80);
        assert_eq!(x_test.nrows(), 20);
        assert_eq!(y_test.len(), 20);
        assert_eq!(x_train.ncols(), features.ncols());
        assert_eq!(x_test.ncols(), features.ncols());
    }
    
    #[test]
    fn test_data_splitter_temporal() {
        let features = Array2::random((50, 3), Uniform::new(-1.0, 1.0));
        let targets = Array1::random(50, Uniform::new(0.0, 1.0));
        
        let split = DataSplit::Temporal { split_point: 40 };
        
        let (x_train, y_train, x_test, y_test) = DataSplitter::split(&split, &features, &targets).unwrap();
        
        assert_eq!(x_train.nrows(), 40);
        assert_eq!(y_train.len(), 40);
        assert_eq!(x_test.nrows(), 10);
        assert_eq!(y_test.len(), 10);
    }
    
    #[test]
    fn test_scaling_methods() {
        assert_eq!(ScalingMethod::Standard, ScalingMethod::Standard);
        assert_ne!(ScalingMethod::Standard, ScalingMethod::MinMax);
        assert_ne!(ScalingMethod::MinMax, ScalingMethod::Robust);
        assert_ne!(ScalingMethod::Robust, ScalingMethod::Unit);
        assert_ne!(ScalingMethod::Unit, ScalingMethod::MaxAbs);
    }
    
    #[test]
    fn test_missing_value_strategies() {
        assert_eq!(MissingValueStrategy::Mean, MissingValueStrategy::Mean);
        assert_ne!(MissingValueStrategy::Mean, MissingValueStrategy::Median);
        assert_eq!(MissingValueStrategy::Constant(1.0), MissingValueStrategy::Constant(1.0));
        assert_ne!(MissingValueStrategy::Constant(1.0), MissingValueStrategy::Constant(2.0));
    }
    
    #[test]
    fn test_feature_importance_methods() {
        assert_eq!(FeatureImportanceMethod::Correlation, FeatureImportanceMethod::Correlation);
        assert_ne!(FeatureImportanceMethod::Correlation, FeatureImportanceMethod::Variance);
        assert_ne!(FeatureImportanceMethod::Variance, FeatureImportanceMethod::Permutation);
    }
}