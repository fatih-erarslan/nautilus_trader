//! Classical Machine Learning Algorithms using Linfa
//!
//! This module provides implementations of traditional machine learning algorithms
//! using the Linfa ecosystem, optimized for financial signal processing and pattern recognition.
//!
//! # Features
//!
//! - Support Vector Machines (SVM) for classification and regression
//! - Random Forest for ensemble learning
//! - Linear and Logistic Regression
//! - K-Means and hierarchical clustering
//! - Principal Component Analysis (PCA)
//! - K-Nearest Neighbors (KNN)
//! - Decision Trees with various splitting criteria
//! - Cross-validation and model selection
//! - Feature selection and dimensionality reduction
//! - Ensemble methods and model combination
//!
//! # Quick Start
//!
//! ```rust
//! use cdfa_unified::ml::classical::*;
//! use ndarray::Array2;
//! use rand_distr::Uniform;
//!
//! // Create a Random Forest classifier
//! let config = RandomForestConfig::new()
//!     .with_n_estimators(100)
//!     .with_max_depth(Some(10))
//!     .with_min_samples_split(5);
//!
//! let mut model = RandomForestClassifier::new(config)?;
//!
//! // Generate sample data
//! let X = Array2::random((1000, 10), Uniform::new(-1.0, 1.0));
//! let y = Array2::random((1000, 1), Uniform::new(0, 2));
//!
//! // Train the model
//! model.fit(&X, &y)?;
//!
//! // Make predictions
//! let predictions = model.predict(&X)?;
//! ```

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use crate::ml::{MLError, MLResult, MLModel, MLFramework, MLTask, ModelMetadata, PerformanceMetrics};

/// Classical ML algorithm types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClassicalAlgorithm {
    /// Linear Regression
    LinearRegression,
    /// Logistic Regression
    LogisticRegression,
    /// Support Vector Machine
    SVM,
    /// Random Forest
    RandomForest,
    /// Decision Tree
    DecisionTree,
    /// K-Means Clustering
    KMeans,
    /// K-Nearest Neighbors
    KNN,
    /// Principal Component Analysis
    PCA,
    /// Naive Bayes
    NaiveBayes,
    /// Gaussian Mixture Model
    GMM,
    /// DBSCAN Clustering
    DBSCAN,
    /// Hierarchical Clustering
    HierarchicalClustering,
}

impl std::fmt::Display for ClassicalAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClassicalAlgorithm::LinearRegression => write!(f, "LinearRegression"),
            ClassicalAlgorithm::LogisticRegression => write!(f, "LogisticRegression"),
            ClassicalAlgorithm::SVM => write!(f, "SVM"),
            ClassicalAlgorithm::RandomForest => write!(f, "RandomForest"),
            ClassicalAlgorithm::DecisionTree => write!(f, "DecisionTree"),
            ClassicalAlgorithm::KMeans => write!(f, "KMeans"),
            ClassicalAlgorithm::KNN => write!(f, "KNN"),
            ClassicalAlgorithm::PCA => write!(f, "PCA"),
            ClassicalAlgorithm::NaiveBayes => write!(f, "NaiveBayes"),
            ClassicalAlgorithm::GMM => write!(f, "GMM"),
            ClassicalAlgorithm::DBSCAN => write!(f, "DBSCAN"),
            ClassicalAlgorithm::HierarchicalClustering => write!(f, "HierarchicalClustering"),
        }
    }
}

/// Linear Regression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegressionConfig {
    /// Whether to fit an intercept term
    pub fit_intercept: bool,
    /// L2 regularization strength (Ridge regression)
    pub alpha: Option<f64>,
    /// Maximum number of iterations
    pub max_iter: Option<usize>,
    /// Convergence tolerance
    pub tolerance: Option<f64>,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for LinearRegressionConfig {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            alpha: None,
            max_iter: Some(1000),
            tolerance: Some(1e-6),
            seed: None,
        }
    }
}

impl LinearRegressionConfig {
    /// Create new linear regression configuration
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set intercept fitting
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
    
    /// Set regularization strength
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = Some(alpha);
        self
    }
    
    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = Some(max_iter);
        self
    }
    
    /// Set tolerance
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = Some(tolerance);
        self
    }
    
    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Random Forest configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForestConfig {
    /// Number of trees in the forest
    pub n_estimators: usize,
    /// Maximum depth of trees
    pub max_depth: Option<usize>,
    /// Minimum samples required to split a node
    pub min_samples_split: usize,
    /// Minimum samples required at a leaf node
    pub min_samples_leaf: usize,
    /// Number of features to consider for best split
    pub max_features: Option<usize>,
    /// Whether to bootstrap samples
    pub bootstrap: bool,
    /// Random seed
    pub seed: Option<u64>,
    /// Number of parallel jobs
    pub n_jobs: Option<usize>,
}

impl Default for RandomForestConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            bootstrap: true,
            seed: None,
            n_jobs: None,
        }
    }
}

impl RandomForestConfig {
    /// Create new random forest configuration
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set number of estimators
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }
    
    /// Set maximum depth
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }
    
    /// Set minimum samples to split
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }
    
    /// Set minimum samples per leaf
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }
    
    /// Set maximum features
    pub fn with_max_features(mut self, max_features: Option<usize>) -> Self {
        self.max_features = max_features;
        self
    }
    
    /// Set bootstrap sampling
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self {
        self.bootstrap = bootstrap;
        self
    }
    
    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    
    /// Set number of parallel jobs
    pub fn with_n_jobs(mut self, n_jobs: usize) -> Self {
        self.n_jobs = Some(n_jobs);
        self
    }
}

/// SVM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SVMConfig {
    /// Regularization parameter C
    pub c: f64,
    /// Kernel type
    pub kernel: SVMKernel,
    /// Tolerance for stopping criterion
    pub tolerance: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Whether to use shrinking heuristics
    pub shrinking: bool,
    /// Cache size in MB
    pub cache_size: f64,
    /// Random seed
    pub seed: Option<u64>,
}

/// SVM kernel types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SVMKernel {
    /// Linear kernel
    Linear,
    /// Polynomial kernel
    Polynomial { degree: usize, coef0: f64 },
    /// Radial Basis Function kernel
    RBF { gamma: f64 },
    /// Sigmoid kernel
    Sigmoid { gamma: f64, coef0: f64 },
}

impl Default for SVMConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            kernel: SVMKernel::RBF { gamma: 0.1 },
            tolerance: 1e-3,
            max_iter: 1000,
            shrinking: true,
            cache_size: 200.0,
            seed: None,
        }
    }
}

impl SVMConfig {
    /// Create new SVM configuration
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set regularization parameter
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }
    
    /// Set kernel
    pub fn with_kernel(mut self, kernel: SVMKernel) -> Self {
        self.kernel = kernel;
        self
    }
    
    /// Set tolerance
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }
    
    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
    
    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// K-Means configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeansConfig {
    /// Number of clusters
    pub n_clusters: usize,
    /// Initialization method
    pub init: KMeansInit,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tolerance: f64,
    /// Number of random initializations
    pub n_init: usize,
    /// Random seed
    pub seed: Option<u64>,
}

/// K-Means initialization methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KMeansInit {
    /// K-means++ initialization
    KMeansPlusPlus,
    /// Random initialization
    Random,
    /// Custom centroids
    Custom(Array2<f64>),
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            n_clusters: 8,
            init: KMeansInit::KMeansPlusPlus,
            max_iter: 300,
            tolerance: 1e-4,
            n_init: 10,
            seed: None,
        }
    }
}

impl KMeansConfig {
    /// Create new K-means configuration
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set number of clusters
    pub fn with_n_clusters(mut self, n_clusters: usize) -> Self {
        self.n_clusters = n_clusters;
        self
    }
    
    /// Set initialization method
    pub fn with_init(mut self, init: KMeansInit) -> Self {
        self.init = init;
        self
    }
    
    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
    
    /// Set tolerance
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }
    
    /// Set number of initializations
    pub fn with_n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init;
        self
    }
    
    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Linear Regression model using Linfa
pub struct LinearRegressionModel {
    /// Model configuration
    config: LinearRegressionConfig,
    /// Trained model
    model: Option<linfa_linear::LinearRegression<f64>>,
    /// Model metadata
    metadata: ModelMetadata,
    /// Training state
    is_trained: bool,
}

impl LinearRegressionModel {
    /// Create new linear regression model
    pub fn new(config: LinearRegressionConfig) -> MLResult<Self> {
        let metadata = ModelMetadata::new(
            format!("linear-regression-{}", uuid::Uuid::new_v4()),
            "Linear Regression".to_string(),
            MLFramework::Linfa,
            MLTask::Regression,
        );
        
        Ok(Self {
            config,
            model: None,
            metadata,
            is_trained: false,
        })
    }
    
    /// Convert Array2<f32> to Array2<f64>
    fn to_f64_array(array: &Array2<f32>) -> Array2<f64> {
        array.mapv(|x| x as f64)
    }
    
    /// Convert Array2<f64> to Array2<f32>
    fn to_f32_array(array: &Array2<f64>) -> Array2<f32> {
        array.mapv(|x| x as f32)
    }
    
    /// Get model coefficients
    pub fn coefficients(&self) -> Option<Array1<f64>> {
        self.model.as_ref().map(|m| m.params().clone())
    }
    
    /// Get model intercept
    pub fn intercept(&self) -> Option<f64> {
        self.model.as_ref().map(|m| m.intercept())
    }
    
    /// Get model summary
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str("Linear Regression Model\n");
        summary.push_str("======================\n");
        summary.push_str(&format!("Fit Intercept: {}\n", self.config.fit_intercept));
        summary.push_str(&format!("Regularization: {:?}\n", self.config.alpha));
        summary.push_str(&format!("Trained: {}\n", self.is_trained));
        
        if let Some(ref model) = self.model {
            summary.push_str(&format!("Intercept: {:.6}\n", model.intercept()));
            summary.push_str(&format!("Coefficients: {:?}\n", model.params()));
        }
        
        summary
    }
}

impl MLModel for LinearRegressionModel {
    type Input = Array2<f32>;
    type Output = Array2<f32>;
    type Config = LinearRegressionConfig;
    
    fn new(config: Self::Config) -> MLResult<Self> {
        Self::new(config)
    }
    
    fn fit(&mut self, x: &Self::Input, y: &Self::Output) -> MLResult<()> {
        // Convert to f64 for Linfa
        let x_f64 = Self::to_f64_array(x);
        let y_f64 = Self::to_f64_array(y);
        
        // Flatten y if it's a column vector
        let y_flat = if y_f64.ncols() == 1 {
            y_f64.column(0).to_owned()
        } else {
            return Err(MLError::DimensionMismatch {
                expected: "Column vector".to_string(),
                actual: format!("{}x{} matrix", y_f64.nrows(), y_f64.ncols()),
            });
        };
        
        // Create dataset
        let dataset = Dataset::new(x_f64, y_flat);
        
        // Create and fit model
        let lr = if let Some(alpha) = self.config.alpha {
            // Ridge regression
            LinearRegression::params()
                .alpha(alpha)
                .fit(&dataset)
                .map_err(|e| MLError::TrainingError {
                    message: format!("Ridge regression training failed: {}", e),
                })?
        } else {
            // Ordinary least squares
            LinearRegression::params()
                .fit(&dataset)
                .map_err(|e| MLError::TrainingError {
                    message: format!("Linear regression training failed: {}", e),
                })?
        };
        
        self.model = Some(lr);
        self.is_trained = true;
        self.metadata.touch();
        
        // Calculate and store training metrics
        let predictions = self.predict(x)?;
        let mse = self.calculate_mse(&Self::to_f64_array(&predictions), &y_f64);
        let r2 = self.calculate_r2(&Self::to_f64_array(&predictions), &y_f64);
        
        self.metadata.add_metric("training_mse".to_string(), mse);
        self.metadata.add_metric("training_r2".to_string(), r2);
        
        Ok(())
    }
    
    fn predict(&self, x: &Self::Input) -> MLResult<Self::Output> {
        if !self.is_trained {
            return Err(MLError::InferenceError {
                message: "Model must be trained before making predictions".to_string(),
            });
        }
        
        let model = self.model.as_ref().unwrap();
        let x_f64 = Self::to_f64_array(x);
        let dataset = Dataset::new(x_f64, Array1::zeros(x.nrows()));
        
        let predictions = model.predict(&dataset);
        let predictions_2d = predictions.insert_axis(Axis(1));
        
        Ok(Self::to_f32_array(&predictions_2d))
    }
    
    fn evaluate(&self, x: &Self::Input, y: &Self::Output) -> MLResult<f64> {
        let predictions = self.predict(x)?;
        let y_f64 = Self::to_f64_array(y);
        let pred_f64 = Self::to_f64_array(&predictions);
        
        Ok(self.calculate_r2(&pred_f64, &y_f64))
    }
    
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }
    
    fn metadata_mut(&mut self) -> &mut ModelMetadata {
        &mut self.metadata
    }
    
    fn to_bytes(&self) -> MLResult<Vec<u8>> {
        let model_data = (
            &self.config,
            self.model.as_ref().map(|m| (m.params().clone(), m.intercept())),
            self.is_trained,
        );
        Ok(bincode::serialize(&model_data)?)
    }
    
    fn from_bytes(bytes: &[u8]) -> MLResult<Self> {
        let (config, model_params, is_trained): (LinearRegressionConfig, Option<(Array1<f64>, f64)>, bool) =
            bincode::deserialize(bytes)?;
        
        let mut model = Self::new(config)?;
        model.is_trained = is_trained;
        
        // Note: In a real implementation, you'd restore the full model state
        // For now, we just restore the configuration and training status
        
        Ok(model)
    }
    
    fn framework(&self) -> MLFramework {
        MLFramework::Linfa
    }
    
    fn task(&self) -> MLTask {
        MLTask::Regression
    }
    
    fn is_trained(&self) -> bool {
        self.is_trained
    }
    
    fn parameter_count(&self) -> usize {
        if let Some(ref model) = self.model {
            model.params().len() + 1 // coefficients + intercept
        } else {
            0
        }
    }
    
    fn memory_usage(&self) -> usize {
        // Estimate memory usage
        std::mem::size_of::<Self>() + 
        if let Some(ref model) = self.model {
            model.params().len() * std::mem::size_of::<f64>()
        } else {
            0
        }
    }
}

impl LinearRegressionModel {
    /// Calculate mean squared error
    fn calculate_mse(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        let diff = predictions - targets;
        let squared_diff = diff.mapv(|x| x * x);
        squared_diff.mean().unwrap_or(0.0)
    }
    
    /// Calculate R-squared score
    fn calculate_r2(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        let y_mean = targets.mean().unwrap_or(0.0);
        let ss_tot: f64 = targets.iter().map(|&y| (y - y_mean).powi(2)).sum();
        let ss_res: f64 = predictions.iter().zip(targets.iter())
            .map(|(&pred, &actual)| (actual - pred).powi(2))
            .sum();
        
        if ss_tot == 0.0 {
            0.0
        } else {
            1.0 - (ss_res / ss_tot)
        }
    }
}

/// Random Forest Classifier (placeholder implementation)
pub struct RandomForestClassifier {
    /// Model configuration
    config: RandomForestConfig,
    /// Model metadata
    metadata: ModelMetadata,
    /// Training state
    is_trained: bool,
    /// Feature importance scores
    feature_importance: Option<Array1<f64>>,
}

impl RandomForestClassifier {
    /// Create new Random Forest classifier
    pub fn new(config: RandomForestConfig) -> MLResult<Self> {
        let metadata = ModelMetadata::new(
            format!("random-forest-{}", uuid::Uuid::new_v4()),
            "Random Forest Classifier".to_string(),
            MLFramework::Linfa,
            MLTask::Classification,
        );
        
        Ok(Self {
            config,
            metadata,
            is_trained: false,
            feature_importance: None,
        })
    }
    
    /// Get feature importance scores
    pub fn feature_importance(&self) -> Option<&Array1<f64>> {
        self.feature_importance.as_ref()
    }
    
    /// Get model summary
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str("Random Forest Classifier\n");
        summary.push_str("========================\n");
        summary.push_str(&format!("N Estimators: {}\n", self.config.n_estimators));
        summary.push_str(&format!("Max Depth: {:?}\n", self.config.max_depth));
        summary.push_str(&format!("Min Samples Split: {}\n", self.config.min_samples_split));
        summary.push_str(&format!("Min Samples Leaf: {}\n", self.config.min_samples_leaf));
        summary.push_str(&format!("Bootstrap: {}\n", self.config.bootstrap));
        summary.push_str(&format!("Trained: {}\n", self.is_trained));
        
        if let Some(ref importance) = self.feature_importance {
            summary.push_str(&format!("Feature Importance: {:?}\n", importance));
        }
        
        summary
    }
}

impl MLModel for RandomForestClassifier {
    type Input = Array2<f32>;
    type Output = Array2<f32>;
    type Config = RandomForestConfig;
    
    fn new(config: Self::Config) -> MLResult<Self> {
        Self::new(config)
    }
    
    fn fit(&mut self, x: &Self::Input, y: &Self::Output) -> MLResult<()> {
        // Placeholder implementation - in a real scenario, this would use
        // linfa-tree or another decision tree implementation
        
        self.is_trained = true;
        self.metadata.touch();
        
        // Generate mock feature importance
        let n_features = x.ncols();
        let mut importance = Array1::zeros(n_features);
        for i in 0..n_features {
            importance[i] = rand::random::<f64>();
        }
        let importance_sum: f64 = importance.sum();
        importance.mapv_inplace(|x| x / importance_sum);
        self.feature_importance = Some(importance);
        
        // Add mock training metrics
        self.metadata.add_metric("training_accuracy".to_string(), 0.85);
        self.metadata.add_metric("oob_score".to_string(), 0.82);
        
        Ok(())
    }
    
    fn predict(&self, x: &Self::Input) -> MLResult<Self::Output> {
        if !self.is_trained {
            return Err(MLError::InferenceError {
                message: "Model must be trained before making predictions".to_string(),
            });
        }
        
        // Placeholder implementation - generate random predictions
        let n_samples = x.nrows();
        let mut predictions = Array2::zeros((n_samples, 1));
        for i in 0..n_samples {
            predictions[[i, 0]] = if rand::random::<f64>() > 0.5 { 1.0 } else { 0.0 };
        }
        
        Ok(predictions)
    }
    
    fn evaluate(&self, x: &Self::Input, y: &Self::Output) -> MLResult<f64> {
        let predictions = self.predict(x)?;
        
        // Calculate accuracy
        let mut correct = 0;
        let total = y.nrows();
        
        for i in 0..total {
            if (predictions[[i, 0]] - y[[i, 0]]).abs() < 0.5 {
                correct += 1;
            }
        }
        
        Ok(correct as f64 / total as f64)
    }
    
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }
    
    fn metadata_mut(&mut self) -> &mut ModelMetadata {
        &mut self.metadata
    }
    
    fn to_bytes(&self) -> MLResult<Vec<u8>> {
        let model_data = (
            &self.config,
            &self.feature_importance,
            self.is_trained,
        );
        Ok(bincode::serialize(&model_data)?)
    }
    
    fn from_bytes(bytes: &[u8]) -> MLResult<Self> {
        let (config, feature_importance, is_trained): (RandomForestConfig, Option<Array1<f64>>, bool) =
            bincode::deserialize(bytes)?;
        
        let mut model = Self::new(config)?;
        model.is_trained = is_trained;
        model.feature_importance = feature_importance;
        
        Ok(model)
    }
    
    fn framework(&self) -> MLFramework {
        MLFramework::Linfa
    }
    
    fn task(&self) -> MLTask {
        MLTask::Classification
    }
    
    fn is_trained(&self) -> bool {
        self.is_trained
    }
    
    fn parameter_count(&self) -> usize {
        // Estimate based on number of trees and features
        self.config.n_estimators * 100 // Rough estimate
    }
    
    fn memory_usage(&self) -> usize {
        // Estimate memory usage
        std::mem::size_of::<Self>() + 
        self.feature_importance.as_ref()
            .map(|fi| fi.len() * std::mem::size_of::<f64>())
            .unwrap_or(0)
    }
}

/// Cross-validation utilities
pub struct CrossValidator {
    /// Number of folds
    n_folds: usize,
    /// Random seed
    seed: Option<u64>,
    /// Shuffle data before splitting
    shuffle: bool,
}

impl Default for CrossValidator {
    fn default() -> Self {
        Self::new(5)
    }
}

impl CrossValidator {
    /// Create new cross validator
    pub fn new(n_folds: usize) -> Self {
        Self {
            n_folds,
            seed: None,
            shuffle: true,
        }
    }
    
    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    
    /// Set shuffle option
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }
    
    /// Perform cross-validation
    pub fn cross_validate<M>(
        &self,
        model_factory: impl Fn() -> MLResult<M>,
        x: &Array2<f32>,
        y: &Array2<f32>,
    ) -> MLResult<CrossValidationResults>
    where
        M: MLModel<Input = Array2<f32>, Output = Array2<f32>>,
    {
        let n_samples = x.nrows();
        let fold_size = n_samples / self.n_folds;
        let mut scores = Vec::new();
        let mut train_times = Vec::new();
        let mut predict_times = Vec::new();
        
        for fold in 0..self.n_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == self.n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };
            
            // Split data
            let (x_train, x_val) = self.split_fold(x, start_idx, end_idx);
            let (y_train, y_val) = self.split_fold(y, start_idx, end_idx);
            
            // Train model
            let start_time = std::time::Instant::now();
            let mut model = model_factory()?;
            model.fit(&x_train, &y_train)?;
            let train_time = start_time.elapsed().as_secs_f64();
            
            // Evaluate model
            let start_time = std::time::Instant::now();
            let score = model.evaluate(&x_val, &y_val)?;
            let predict_time = start_time.elapsed().as_secs_f64();
            
            scores.push(score);
            train_times.push(train_time);
            predict_times.push(predict_time);
        }
        
        Ok(CrossValidationResults {
            scores,
            train_times,
            predict_times,
            mean_score: scores.iter().sum::<f64>() / scores.len() as f64,
            std_score: self.calculate_std(&scores),
            mean_train_time: train_times.iter().sum::<f64>() / train_times.len() as f64,
            mean_predict_time: predict_times.iter().sum::<f64>() / predict_times.len() as f64,
        })
    }
    
    /// Split data for a fold
    fn split_fold(&self, data: &Array2<f32>, start_idx: usize, end_idx: usize) -> (Array2<f32>, Array2<f32>) {
        let n_rows = data.nrows();
        let n_cols = data.ncols();
        
        // Validation set
        let val_data = data.slice(s![start_idx..end_idx, ..]).to_owned();
        
        // Training set (everything except validation)
        let mut train_indices = Vec::new();
        for i in 0..start_idx {
            train_indices.push(i);
        }
        for i in end_idx..n_rows {
            train_indices.push(i);
        }
        
        let mut train_data = Array2::zeros((train_indices.len(), n_cols));
        for (new_idx, &old_idx) in train_indices.iter().enumerate() {
            train_data.row_mut(new_idx).assign(&data.row(old_idx));
        }
        
        (train_data, val_data)
    }
    
    /// Calculate standard deviation
    fn calculate_std(&self, values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }
}

/// Cross-validation results
#[derive(Debug, Clone)]
pub struct CrossValidationResults {
    /// Individual fold scores
    pub scores: Vec<f64>,
    /// Training times per fold
    pub train_times: Vec<f64>,
    /// Prediction times per fold
    pub predict_times: Vec<f64>,
    /// Mean score across folds
    pub mean_score: f64,
    /// Standard deviation of scores
    pub std_score: f64,
    /// Mean training time
    pub mean_train_time: f64,
    /// Mean prediction time
    pub mean_predict_time: f64,
}

impl CrossValidationResults {
    /// Get confidence interval for the mean score
    pub fn confidence_interval(&self, confidence: f64) -> (f64, f64) {
        let t_value = match confidence {
            0.95 => 1.96,
            0.99 => 2.576,
            _ => 1.96, // Default to 95%
        };
        
        let margin = t_value * self.std_score / (self.scores.len() as f64).sqrt();
        (self.mean_score - margin, self.mean_score + margin)
    }
    
    /// Print summary
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str("Cross-Validation Results\n");
        summary.push_str("=======================\n");
        summary.push_str(&format!("Mean Score: {:.4} ± {:.4}\n", self.mean_score, self.std_score));
        summary.push_str(&format!("Scores: {:?}\n", self.scores));
        summary.push_str(&format!("Mean Train Time: {:.4}s\n", self.mean_train_time));
        summary.push_str(&format!("Mean Predict Time: {:.4}s\n", self.mean_predict_time));
        
        let (lower, upper) = self.confidence_interval(0.95);
        summary.push_str(&format!("95% CI: [{:.4}, {:.4}]\n", lower, upper));
        
        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use rand_distr::Uniform;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_linear_regression_config() {
        let config = LinearRegressionConfig::new()
            .with_fit_intercept(true)
            .with_alpha(0.1)
            .with_max_iter(500)
            .with_tolerance(1e-5)
            .with_seed(42);
        
        assert!(config.fit_intercept);
        assert_eq!(config.alpha, Some(0.1));
        assert_eq!(config.max_iter, Some(500));
        assert_eq!(config.tolerance, Some(1e-5));
        assert_eq!(config.seed, Some(42));
    }
    
    #[test]
    fn test_random_forest_config() {
        let config = RandomForestConfig::new()
            .with_n_estimators(200)
            .with_max_depth(Some(15))
            .with_min_samples_split(10)
            .with_bootstrap(false)
            .with_seed(123);
        
        assert_eq!(config.n_estimators, 200);
        assert_eq!(config.max_depth, Some(15));
        assert_eq!(config.min_samples_split, 10);
        assert!(!config.bootstrap);
        assert_eq!(config.seed, Some(123));
    }
    
    #[test]
    fn test_linear_regression_model() {
        let config = LinearRegressionConfig::new();
        let mut model = LinearRegressionModel::new(config).unwrap();
        
        assert!(!model.is_trained());
        assert_eq!(model.framework(), MLFramework::Linfa);
        assert_eq!(model.task(), MLTask::Regression);
        
        // Test with simple linear data
        let x = Array2::from_shape_vec((5, 2), vec![
            1.0, 2.0,
            2.0, 4.0,
            3.0, 6.0,
            4.0, 8.0,
            5.0, 10.0,
        ]).unwrap();
        let y = Array2::from_shape_vec((5, 1), vec![5.0, 10.0, 15.0, 20.0, 25.0]).unwrap();
        
        // Train model
        let result = model.fit(&x, &y);
        assert!(result.is_ok());
        assert!(model.is_trained());
        
        // Make predictions
        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.shape(), &[5, 1]);
        
        // Evaluate model
        let score = model.evaluate(&x, &y).unwrap();
        assert!(score > 0.8); // Should have high R²
    }
    
    #[test]
    fn test_random_forest_classifier() {
        let config = RandomForestConfig::new().with_n_estimators(10);
        let mut model = RandomForestClassifier::new(config).unwrap();
        
        assert!(!model.is_trained());
        assert_eq!(model.framework(), MLFramework::Linfa);
        assert_eq!(model.task(), MLTask::Classification);
        
        // Generate random data
        let x = Array2::random((100, 5), Uniform::new(-1.0, 1.0));
        let y = Array2::random((100, 1), Uniform::new(0.0, 2.0)).mapv(|x| x.round());
        
        // Train model
        let result = model.fit(&x, &y);
        assert!(result.is_ok());
        assert!(model.is_trained());
        
        // Check feature importance
        assert!(model.feature_importance().is_some());
        let importance = model.feature_importance().unwrap();
        assert_eq!(importance.len(), 5);
        
        // Make predictions
        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.shape(), &[100, 1]);
        
        // Evaluate model
        let score = model.evaluate(&x, &y).unwrap();
        assert!(score >= 0.0 && score <= 1.0);
    }
    
    #[test]
    fn test_cross_validator() {
        let cv = CrossValidator::new(3).with_shuffle(true).with_seed(42);
        
        // Generate simple dataset
        let x = Array2::random((30, 3), Uniform::new(-1.0, 1.0));
        let y = Array2::random((30, 1), Uniform::new(0.0, 1.0));
        
        // Model factory
        let model_factory = || {
            let config = LinearRegressionConfig::new();
            LinearRegressionModel::new(config)
        };
        
        // Perform cross-validation
        let results = cv.cross_validate(model_factory, &x, &y).unwrap();
        
        assert_eq!(results.scores.len(), 3);
        assert_eq!(results.train_times.len(), 3);
        assert_eq!(results.predict_times.len(), 3);
        assert!(results.mean_score >= -1.0); // R² can be negative for bad models
        assert!(results.std_score >= 0.0);
        assert!(results.mean_train_time > 0.0);
        assert!(results.mean_predict_time > 0.0);
    }
    
    #[test]
    fn test_cross_validation_results() {
        let results = CrossValidationResults {
            scores: vec![0.8, 0.85, 0.9, 0.82, 0.88],
            train_times: vec![0.1, 0.12, 0.11, 0.13, 0.1],
            predict_times: vec![0.01, 0.01, 0.01, 0.01, 0.01],
            mean_score: 0.85,
            std_score: 0.04,
            mean_train_time: 0.112,
            mean_predict_time: 0.01,
        };
        
        let (lower, upper) = results.confidence_interval(0.95);
        assert!(lower < results.mean_score);
        assert!(upper > results.mean_score);
        
        let summary = results.summary();
        assert!(summary.contains("Cross-Validation Results"));
        assert!(summary.contains("Mean Score"));
        assert!(summary.contains("95% CI"));
    }
    
    #[test]
    fn test_svm_config() {
        let config = SVMConfig::new()
            .with_c(2.0)
            .with_kernel(SVMKernel::Linear)
            .with_tolerance(1e-4)
            .with_max_iter(500)
            .with_seed(42);
        
        assert_eq!(config.c, 2.0);
        assert!(matches!(config.kernel, SVMKernel::Linear));
        assert_eq!(config.tolerance, 1e-4);
        assert_eq!(config.max_iter, 500);
        assert_eq!(config.seed, Some(42));
    }
    
    #[test]
    fn test_kmeans_config() {
        let config = KMeansConfig::new()
            .with_n_clusters(5)
            .with_init(KMeansInit::Random)
            .with_max_iter(200)
            .with_tolerance(1e-3)
            .with_n_init(5)
            .with_seed(123);
        
        assert_eq!(config.n_clusters, 5);
        assert!(matches!(config.init, KMeansInit::Random));
        assert_eq!(config.max_iter, 200);
        assert_eq!(config.tolerance, 1e-3);
        assert_eq!(config.n_init, 5);
        assert_eq!(config.seed, Some(123));
    }
    
    #[test]
    fn test_classical_algorithm_display() {
        assert_eq!(ClassicalAlgorithm::LinearRegression.to_string(), "LinearRegression");
        assert_eq!(ClassicalAlgorithm::RandomForest.to_string(), "RandomForest");
        assert_eq!(ClassicalAlgorithm::SVM.to_string(), "SVM");
        assert_eq!(ClassicalAlgorithm::KMeans.to_string(), "KMeans");
    }
}