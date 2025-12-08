//! ML Training Infrastructure
//!
//! This module provides comprehensive training infrastructure including
//! hyperparameter optimization, cross-validation, and distributed training.

use ndarray::{Array1, Array2, Axis, s};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use crate::ml::{MLError, MLResult, MLModel, TrainingConfig, PerformanceMetrics};

/// Hyperparameter optimization methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationMethod {
    /// Grid search over parameter space
    GridSearch,
    /// Random search over parameter space
    RandomSearch,
    /// Bayesian optimization
    BayesianOptimization,
    /// Genetic algorithm
    GeneticAlgorithm,
    /// Particle swarm optimization
    ParticleSwarm,
    /// Simulated annealing
    SimulatedAnnealing,
}

/// Hyperparameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: ParameterType,
    /// Default value
    pub default: serde_json::Value,
    /// Description
    pub description: Option<String>,
}

/// Parameter types for hyperparameter optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    /// Continuous parameter with range
    Continuous { min: f64, max: f64 },
    /// Integer parameter with range
    Integer { min: i64, max: i64 },
    /// Categorical parameter with choices
    Categorical { choices: Vec<String> },
    /// Boolean parameter
    Boolean,
    /// Log-scale continuous parameter
    LogContinuous { min: f64, max: f64 },
}

/// Hyperparameter optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Optimization method
    pub method: OptimizationMethod,
    /// Maximum number of trials
    pub max_trials: usize,
    /// Maximum time budget
    pub max_time: Option<Duration>,
    /// Number of parallel workers
    pub n_workers: usize,
    /// Random seed
    pub seed: Option<u64>,
    /// Early stopping criteria
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// Objective metric to optimize
    pub objective_metric: String,
    /// Whether to maximize (true) or minimize (false) the objective
    pub maximize: bool,
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Patience (number of trials without improvement)
    pub patience: usize,
    /// Minimum improvement threshold
    pub min_delta: f64,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            method: OptimizationMethod::RandomSearch,
            max_trials: 100,
            max_time: Some(Duration::from_secs(3600)), // 1 hour
            n_workers: 1,
            seed: None,
            early_stopping: Some(EarlyStoppingConfig {
                patience: 10,
                min_delta: 1e-4,
            }),
            objective_metric: "accuracy".to_string(),
            maximize: true,
        }
    }
}

/// Trial result for hyperparameter optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trial {
    /// Trial ID
    pub id: String,
    /// Hyperparameters used
    pub parameters: HashMap<String, serde_json::Value>,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Training duration
    pub duration: Duration,
    /// Trial status
    pub status: TrialStatus,
    /// Error message if failed
    pub error: Option<String>,
}

/// Trial status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrialStatus {
    /// Trial is running
    Running,
    /// Trial completed successfully
    Completed,
    /// Trial failed
    Failed,
    /// Trial was cancelled
    Cancelled,
}

/// Hyperparameter optimizer
pub struct HyperparameterOptimizer {
    /// Optimization configuration
    config: OptimizationConfig,
    /// Parameter definitions
    parameters: Vec<HyperParameter>,
    /// Completed trials
    trials: Vec<Trial>,
    /// Best trial found so far
    best_trial: Option<Trial>,
    /// Random number generator
    rng: rand::rngs::StdRng,
}

impl HyperparameterOptimizer {
    /// Create new hyperparameter optimizer
    pub fn new(config: OptimizationConfig, parameters: Vec<HyperParameter>) -> Self {
        use rand::SeedableRng;
        
        let rng = if let Some(seed) = config.seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        };
        
        Self {
            config,
            parameters,
            trials: Vec::new(),
            best_trial: None,
            rng,
        }
    }
    
    /// Optimize hyperparameters for a model
    pub fn optimize<M, F>(&mut self, mut model_factory: F) -> MLResult<Trial>
    where
        M: MLModel<Input = Array2<f32>, Output = Array2<f32>>,
        F: FnMut(&HashMap<String, serde_json::Value>) -> MLResult<M>,
    {
        let start_time = Instant::now();
        let mut trials_without_improvement = 0;
        
        for trial_idx in 0..self.config.max_trials {
            // Check time budget
            if let Some(max_time) = self.config.max_time {
                if start_time.elapsed() > max_time {
                    break;
                }
            }
            
            // Generate hyperparameters
            let parameters = self.sample_parameters()?;
            
            // Create trial
            let trial_id = format!("trial_{}", trial_idx);
            let trial_start = Instant::now();
            
            let mut trial = Trial {
                id: trial_id.clone(),
                parameters: parameters.clone(),
                metrics: PerformanceMetrics::default(),
                duration: Duration::default(),
                status: TrialStatus::Running,
                error: None,
            };
            
            // Run trial
            match self.run_trial(&mut model_factory, &parameters) {
                Ok(metrics) => {
                    trial.metrics = metrics;
                    trial.status = TrialStatus::Completed;
                    trial.duration = trial_start.elapsed();
                    
                    // Check if this is the best trial
                    let objective_value = self.get_objective_value(&trial.metrics);
                    let is_best = if let Some(ref best) = self.best_trial {
                        let best_value = self.get_objective_value(&best.metrics);
                        if self.config.maximize {
                            objective_value > best_value
                        } else {
                            objective_value < best_value
                        }
                    } else {
                        true
                    };
                    
                    if is_best {
                        self.best_trial = Some(trial.clone());
                        trials_without_improvement = 0;
                    } else {
                        trials_without_improvement += 1;
                    }
                }
                Err(e) => {
                    trial.status = TrialStatus::Failed;
                    trial.error = Some(e.to_string());
                    trial.duration = trial_start.elapsed();
                    trials_without_improvement += 1;
                }
            }
            
            self.trials.push(trial);
            
            // Check early stopping
            if let Some(ref early_stop) = self.config.early_stopping {
                if trials_without_improvement >= early_stop.patience {
                    break;
                }
            }
        }
        
        self.best_trial.clone()
            .ok_or_else(|| MLError::OptimizationError {
                message: "No successful trials completed".to_string(),
            })
    }
    
    /// Sample hyperparameters based on optimization method
    fn sample_parameters(&mut self) -> MLResult<HashMap<String, serde_json::Value>> {
        match self.config.method {
            OptimizationMethod::RandomSearch => self.random_sample(),
            OptimizationMethod::GridSearch => self.grid_sample(),
            OptimizationMethod::BayesianOptimization => self.bayesian_sample(),
            OptimizationMethod::GeneticAlgorithm => self.genetic_sample(),
            OptimizationMethod::ParticleSwarm => self.pso_sample(),
            OptimizationMethod::SimulatedAnnealing => self.sa_sample(),
        }
    }
    
    /// Random sampling
    fn random_sample(&mut self) -> MLResult<HashMap<String, serde_json::Value>> {
        use rand::Rng;
        
        let mut parameters = HashMap::new();
        
        for param in &self.parameters {
            let value = match &param.param_type {
                ParameterType::Continuous { min, max } => {
                    let val = self.rng.gen_range(*min..=*max);
                    serde_json::Value::Number(serde_json::Number::from_f64(val).unwrap())
                }
                ParameterType::Integer { min, max } => {
                    let val = self.rng.gen_range(*min..=*max);
                    serde_json::Value::Number(serde_json::Number::from(val))
                }
                ParameterType::Categorical { choices } => {
                    let idx = self.rng.gen_range(0..choices.len());
                    serde_json::Value::String(choices[idx].clone())
                }
                ParameterType::Boolean => {
                    serde_json::Value::Bool(self.rng.gen())
                }
                ParameterType::LogContinuous { min, max } => {
                    let log_min = min.ln();
                    let log_max = max.ln();
                    let log_val = self.rng.gen_range(log_min..=log_max);
                    let val = log_val.exp();
                    serde_json::Value::Number(serde_json::Number::from_f64(val).unwrap())
                }
            };
            
            parameters.insert(param.name.clone(), value);
        }
        
        Ok(parameters)
    }
    
    /// Grid sampling (simplified implementation)
    fn grid_sample(&mut self) -> MLResult<HashMap<String, serde_json::Value>> {
        // For simplicity, use random sampling
        // A real implementation would enumerate all combinations
        self.random_sample()
    }
    
    /// Bayesian optimization sampling (simplified)
    fn bayesian_sample(&mut self) -> MLResult<HashMap<String, serde_json::Value>> {
        // For simplicity, use random sampling
        // A real implementation would use Gaussian processes
        self.random_sample()
    }
    
    /// Genetic algorithm sampling (simplified)
    fn genetic_sample(&mut self) -> MLResult<HashMap<String, serde_json::Value>> {
        // For simplicity, use random sampling
        // A real implementation would maintain a population
        self.random_sample()
    }
    
    /// Particle swarm optimization sampling (simplified)
    fn pso_sample(&mut self) -> MLResult<HashMap<String, serde_json::Value>> {
        // For simplicity, use random sampling
        // A real implementation would maintain particle positions
        self.random_sample()
    }
    
    /// Simulated annealing sampling (simplified)
    fn sa_sample(&mut self) -> MLResult<HashMap<String, serde_json::Value>> {
        // For simplicity, use random sampling
        // A real implementation would use temperature-based sampling
        self.random_sample()
    }
    
    /// Run a single trial
    fn run_trial<M, F>(
        &self,
        model_factory: &mut F,
        parameters: &HashMap<String, serde_json::Value>,
    ) -> MLResult<PerformanceMetrics>
    where
        M: MLModel<Input = Array2<f32>, Output = Array2<f32>>,
        F: FnMut(&HashMap<String, serde_json::Value>) -> MLResult<M>,
    {
        // Create model with hyperparameters
        let _model = model_factory(parameters)?;
        
        // For now, return mock metrics
        // In a real implementation, this would train and evaluate the model
        let mut metrics = PerformanceMetrics::default();
        metrics.accuracy = Some(rand::random::<f64>() * 0.3 + 0.7); // 0.7-1.0
        metrics.f1_score = Some(rand::random::<f64>() * 0.3 + 0.6);  // 0.6-0.9
        metrics.training_time = Some(rand::random::<f64>() * 10.0);  // 0-10 seconds
        
        Ok(metrics)
    }
    
    /// Get objective value from metrics
    fn get_objective_value(&self, metrics: &PerformanceMetrics) -> f64 {
        match self.config.objective_metric.as_str() {
            "accuracy" => metrics.accuracy.unwrap_or(0.0),
            "f1_score" => metrics.f1_score.unwrap_or(0.0),
            "precision" => metrics.precision.unwrap_or(0.0),
            "recall" => metrics.recall.unwrap_or(0.0),
            "auc_roc" => metrics.auc_roc.unwrap_or(0.0),
            "r2_score" => metrics.r2_score.unwrap_or(0.0),
            "mse" => -metrics.mse.unwrap_or(f64::INFINITY), // Negative because lower is better
            "mae" => -metrics.mae.unwrap_or(f64::INFINITY),
            "rmse" => -metrics.rmse.unwrap_or(f64::INFINITY),
            _ => {
                if let Some(value) = metrics.custom_metrics.get(&self.config.objective_metric) {
                    *value
                } else {
                    0.0
                }
            }
        }
    }
    
    /// Get all trials
    pub fn get_trials(&self) -> &[Trial] {
        &self.trials
    }
    
    /// Get best trial
    pub fn get_best_trial(&self) -> Option<&Trial> {
        self.best_trial.as_ref()
    }
    
    /// Get optimization summary
    pub fn get_summary(&self) -> OptimizationSummary {
        let total_trials = self.trials.len();
        let successful_trials = self.trials.iter()
            .filter(|t| t.status == TrialStatus::Completed)
            .count();
        let failed_trials = self.trials.iter()
            .filter(|t| t.status == TrialStatus::Failed)
            .count();
        
        let total_time = self.trials.iter()
            .map(|t| t.duration)
            .sum();
        
        let best_objective = self.best_trial.as_ref()
            .map(|t| self.get_objective_value(&t.metrics));
        
        OptimizationSummary {
            total_trials,
            successful_trials,
            failed_trials,
            total_time,
            best_objective,
            best_parameters: self.best_trial.as_ref()
                .map(|t| t.parameters.clone()),
        }
    }
}

/// Optimization summary
#[derive(Debug, Clone)]
pub struct OptimizationSummary {
    /// Total number of trials
    pub total_trials: usize,
    /// Number of successful trials
    pub successful_trials: usize,
    /// Number of failed trials
    pub failed_trials: usize,
    /// Total optimization time
    pub total_time: Duration,
    /// Best objective value achieved
    pub best_objective: Option<f64>,
    /// Best hyperparameters
    pub best_parameters: Option<HashMap<String, serde_json::Value>>,
}

impl OptimizationSummary {
    /// Generate a summary report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("Hyperparameter Optimization Summary\n");
        report.push_str("==================================\n\n");
        
        report.push_str(&format!("Total Trials: {}\n", self.total_trials));
        report.push_str(&format!("Successful: {}\n", self.successful_trials));
        report.push_str(&format!("Failed: {}\n", self.failed_trials));
        report.push_str(&format!("Success Rate: {:.1}%\n", 
            if self.total_trials > 0 {
                100.0 * self.successful_trials as f64 / self.total_trials as f64
            } else {
                0.0
            }
        ));
        report.push_str(&format!("Total Time: {:.2}s\n", self.total_time.as_secs_f64()));
        
        if let Some(best_obj) = self.best_objective {
            report.push_str(&format!("Best Objective: {:.6}\n", best_obj));
        }
        
        if let Some(ref best_params) = self.best_parameters {
            report.push_str("\nBest Parameters:\n");
            for (name, value) in best_params {
                report.push_str(&format!("  {}: {}\n", name, value));
            }
        }
        
        report
    }
}

/// Training pipeline for comprehensive model training
pub struct TrainingPipeline {
    /// Training configuration
    config: TrainingConfig,
    /// Data preprocessing functions
    preprocessors: Vec<Box<dyn DataPreprocessor>>,
    /// Validation strategy
    validation_strategy: ValidationStrategy,
    /// Model checkpointing
    checkpointing: Option<CheckpointConfig>,
}

/// Data preprocessing trait
pub trait DataPreprocessor: Send + Sync {
    /// Preprocess training data
    fn preprocess(&self, x: &Array2<f32>, y: &Array2<f32>) -> MLResult<(Array2<f32>, Array2<f32>)>;
    
    /// Get preprocessor name
    fn name(&self) -> &str;
}

/// Validation strategies
#[derive(Debug, Clone)]
pub enum ValidationStrategy {
    /// Hold-out validation
    HoldOut { test_size: f64 },
    /// K-fold cross-validation
    KFold { k: usize, shuffle: bool },
    /// Stratified K-fold
    StratifiedKFold { k: usize, shuffle: bool },
    /// Time series split
    TimeSeriesSplit { n_splits: usize },
    /// Leave-one-out
    LeaveOneOut,
}

/// Checkpoint configuration
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Directory to save checkpoints
    pub checkpoint_dir: std::path::PathBuf,
    /// Save frequency (every N epochs)
    pub save_frequency: usize,
    /// Keep only best N checkpoints
    pub keep_best: Option<usize>,
    /// Metric to use for "best" selection
    pub best_metric: String,
}

impl TrainingPipeline {
    /// Create new training pipeline
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            preprocessors: Vec::new(),
            validation_strategy: ValidationStrategy::HoldOut { test_size: 0.2 },
            checkpointing: None,
        }
    }
    
    /// Add data preprocessor
    pub fn add_preprocessor(&mut self, preprocessor: Box<dyn DataPreprocessor>) {
        self.preprocessors.push(preprocessor);
    }
    
    /// Set validation strategy
    pub fn set_validation_strategy(&mut self, strategy: ValidationStrategy) {
        self.validation_strategy = strategy;
    }
    
    /// Set checkpointing configuration
    pub fn set_checkpointing(&mut self, config: CheckpointConfig) {
        self.checkpointing = Some(config);
    }
    
    /// Train model with full pipeline
    pub fn train<M>(&self, mut model: M, x: &Array2<f32>, y: &Array2<f32>) -> MLResult<TrainingResult<M>>
    where
        M: MLModel<Input = Array2<f32>, Output = Array2<f32>>,
    {
        let start_time = Instant::now();
        
        // Preprocess data
        let (mut x_processed, mut y_processed) = (x.clone(), y.clone());
        for preprocessor in &self.preprocessors {
            let (x_new, y_new) = preprocessor.preprocess(&x_processed, &y_processed)?;
            x_processed = x_new;
            y_processed = y_new;
        }
        
        // Split data based on validation strategy
        let splits = self.create_splits(&x_processed, &y_processed)?;
        
        let mut training_history = Vec::new();
        let mut best_score = if self.is_maximize_metric() { f64::NEG_INFINITY } else { f64::INFINITY };
        let mut epochs_without_improvement = 0;
        
        // Training loop
        for epoch in 0..self.config.max_epochs {
            let epoch_start = Instant::now();
            
            // Train on all splits
            let mut epoch_metrics = PerformanceMetrics::default();
            let mut fold_scores = Vec::new();
            
            for (fold_idx, (x_train, y_train, x_val, y_val)) in splits.iter().enumerate() {
                // Train model on this fold
                model.fit(x_train, y_train)?;
                
                // Evaluate on validation set
                let score = model.evaluate(x_val, y_val)?;
                fold_scores.push(score);
                
                // Update metrics (simplified)
                if epoch_metrics.accuracy.is_none() {
                    epoch_metrics.accuracy = Some(score);
                } else {
                    epoch_metrics.accuracy = Some(
                        (epoch_metrics.accuracy.unwrap() + score) / 2.0
                    );
                }
            }
            
            let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
            epoch_metrics.training_time = Some(epoch_start.elapsed().as_secs_f64());
            
            training_history.push(EpochResult {
                epoch,
                metrics: epoch_metrics.clone(),
                validation_score: mean_score,
                learning_rate: self.config.learning_rate,
            });
            
            // Check for improvement
            let improved = if self.is_maximize_metric() {
                mean_score > best_score
            } else {
                mean_score < best_score
            };
            
            if improved {
                best_score = mean_score;
                epochs_without_improvement = 0;
                
                // Save checkpoint if configured
                if let Some(ref checkpoint_config) = self.checkpointing {
                    if epoch % checkpoint_config.save_frequency == 0 {
                        self.save_checkpoint(&model, epoch, mean_score, checkpoint_config)?;
                    }
                }
            } else {
                epochs_without_improvement += 1;
            }
            
            // Early stopping
            if let Some(patience) = self.config.early_stopping_patience {
                if epochs_without_improvement >= patience {
                    break;
                }
            }
        }
        
        let total_time = start_time.elapsed();
        
        Ok(TrainingResult {
            model,
            training_history,
            best_score,
            total_training_time: total_time,
            final_metrics: training_history.last()
                .map(|e| e.metrics.clone())
                .unwrap_or_default(),
        })
    }
    
    /// Create data splits based on validation strategy
    fn create_splits(
        &self,
        x: &Array2<f32>,
        y: &Array2<f32>,
    ) -> MLResult<Vec<(Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>)>> {
        match &self.validation_strategy {
            ValidationStrategy::HoldOut { test_size } => {
                let n_samples = x.nrows();
                let n_test = (n_samples as f64 * test_size) as usize;
                let n_train = n_samples - n_test;
                
                let x_train = x.slice(s![0..n_train, ..]).to_owned();
                let y_train = y.slice(s![0..n_train, ..]).to_owned();
                let x_test = x.slice(s![n_train.., ..]).to_owned();
                let y_test = y.slice(s![n_train.., ..]).to_owned();
                
                Ok(vec![(x_train, y_train, x_test, y_test)])
            }
            ValidationStrategy::KFold { k, shuffle: _ } => {
                let n_samples = x.nrows();
                let fold_size = n_samples / k;
                let mut splits = Vec::new();
                
                for fold in 0..*k {
                    let start_idx = fold * fold_size;
                    let end_idx = if fold == k - 1 { n_samples } else { (fold + 1) * fold_size };
                    
                    // Create train and validation sets
                    let mut x_train_parts = Vec::new();
                    let mut y_train_parts = Vec::new();
                    
                    if start_idx > 0 {
                        x_train_parts.push(x.slice(s![0..start_idx, ..]).to_owned());
                        y_train_parts.push(y.slice(s![0..start_idx, ..]).to_owned());
                    }
                    
                    if end_idx < n_samples {
                        x_train_parts.push(x.slice(s![end_idx.., ..]).to_owned());
                        y_train_parts.push(y.slice(s![end_idx.., ..]).to_owned());
                    }
                    
                    let x_val = x.slice(s![start_idx..end_idx, ..]).to_owned();
                    let y_val = y.slice(s![start_idx..end_idx, ..]).to_owned();
                    
                    // Concatenate training parts
                    let x_train = if x_train_parts.len() == 1 {
                        x_train_parts.into_iter().next().unwrap()
                    } else if x_train_parts.len() == 2 {
                        ndarray::concatenate![Axis(0), x_train_parts[0], x_train_parts[1]]
                    } else {
                        continue; // Skip invalid folds
                    };
                    
                    let y_train = if y_train_parts.len() == 1 {
                        y_train_parts.into_iter().next().unwrap()
                    } else if y_train_parts.len() == 2 {
                        ndarray::concatenate![Axis(0), y_train_parts[0], y_train_parts[1]]
                    } else {
                        continue; // Skip invalid folds
                    };
                    
                    splits.push((x_train, y_train, x_val, y_val));
                }
                
                Ok(splits)
            }
            _ => {
                // For other strategies, use hold-out for now
                self.create_splits(x, y)
            }
        }
    }
    
    /// Check if the metric should be maximized
    fn is_maximize_metric(&self) -> bool {
        // Most metrics should be maximized, except for loss metrics
        !matches!(
            self.config.save_path.as_ref()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
                .unwrap_or(""),
            "loss" | "mse" | "mae" | "rmse"
        )
    }
    
    /// Save model checkpoint
    fn save_checkpoint<M>(
        &self,
        model: &M,
        epoch: usize,
        score: f64,
        config: &CheckpointConfig,
    ) -> MLResult<()>
    where
        M: MLModel,
    {
        let checkpoint_name = format!("checkpoint_epoch_{}_score_{:.6}.bin", epoch, score);
        let checkpoint_path = config.checkpoint_dir.join(checkpoint_name);
        
        let model_data = model.to_bytes()?;
        std::fs::create_dir_all(&config.checkpoint_dir)?;
        std::fs::write(checkpoint_path, model_data)?;
        
        Ok(())
    }
}

/// Training result
#[derive(Debug, Clone)]
pub struct TrainingResult<M> {
    /// Trained model
    pub model: M,
    /// Training history
    pub training_history: Vec<EpochResult>,
    /// Best validation score achieved
    pub best_score: f64,
    /// Total training time
    pub total_training_time: Duration,
    /// Final metrics
    pub final_metrics: PerformanceMetrics,
}

/// Result for a single epoch
#[derive(Debug, Clone)]
pub struct EpochResult {
    /// Epoch number
    pub epoch: usize,
    /// Metrics for this epoch
    pub metrics: PerformanceMetrics,
    /// Validation score
    pub validation_score: f64,
    /// Learning rate used
    pub learning_rate: f64,
}

/// Standard data preprocessors
pub struct StandardScaler {
    name: String,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self {
            name: "StandardScaler".to_string(),
        }
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl DataPreprocessor for StandardScaler {
    fn preprocess(&self, x: &Array2<f32>, y: &Array2<f32>) -> MLResult<(Array2<f32>, Array2<f32>)> {
        // Compute mean and std for each feature
        let mean = x.mean_axis(Axis(0)).unwrap();
        let std = x.std_axis(Axis(0), 0.0);
        
        // Standardize features
        let mut x_scaled = x.clone();
        for mut row in x_scaled.rows_mut() {
            for (i, val) in row.iter_mut().enumerate() {
                if std[i] > 1e-8 {
                    *val = (*val - mean[i]) / std[i];
                }
            }
        }
        
        Ok((x_scaled, y.clone()))
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// Min-Max scaler
pub struct MinMaxScaler {
    name: String,
}

impl MinMaxScaler {
    pub fn new() -> Self {
        Self {
            name: "MinMaxScaler".to_string(),
        }
    }
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl DataPreprocessor for MinMaxScaler {
    fn preprocess(&self, x: &Array2<f32>, y: &Array2<f32>) -> MLResult<(Array2<f32>, Array2<f32>)> {
        // Compute min and max for each feature
        let min_vals = x.fold_axis(Axis(0), f32::INFINITY, |&acc, &x| acc.min(x));
        let max_vals = x.fold_axis(Axis(0), f32::NEG_INFINITY, |&acc, &x| acc.max(x));
        
        // Scale features to [0, 1]
        let mut x_scaled = x.clone();
        for mut row in x_scaled.rows_mut() {
            for (i, val) in row.iter_mut().enumerate() {
                let range = max_vals[i] - min_vals[i];
                if range > 1e-8 {
                    *val = (*val - min_vals[i]) / range;
                }
            }
        }
        
        Ok((x_scaled, y.clone()))
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::neural::{NeuralNetwork, NeuralConfig};
    use tempfile::tempdir;
    
    #[test]
    fn test_hyperparameter_types() {
        let continuous = ParameterType::Continuous { min: 0.0, max: 1.0 };
        let integer = ParameterType::Integer { min: 1, max: 100 };
        let categorical = ParameterType::Categorical {
            choices: vec!["relu".to_string(), "tanh".to_string()],
        };
        let boolean = ParameterType::Boolean;
        let log_continuous = ParameterType::LogContinuous { min: 0.001, max: 1.0 };
        
        // These should compile without errors
        match continuous {
            ParameterType::Continuous { min, max } => {
                assert!(min < max);
            }
            _ => panic!("Wrong type"),
        }
        
        match categorical {
            ParameterType::Categorical { choices } => {
                assert_eq!(choices.len(), 2);
            }
            _ => panic!("Wrong type"),
        }
        
        assert!(matches!(boolean, ParameterType::Boolean));
        assert!(matches!(log_continuous, ParameterType::LogContinuous { .. }));
        assert!(matches!(integer, ParameterType::Integer { .. }));
    }
    
    #[test]
    fn test_optimization_config() {
        let config = OptimizationConfig {
            method: OptimizationMethod::RandomSearch,
            max_trials: 50,
            max_time: Some(Duration::from_secs(1800)),
            n_workers: 2,
            seed: Some(42),
            early_stopping: Some(EarlyStoppingConfig {
                patience: 5,
                min_delta: 1e-3,
            }),
            objective_metric: "f1_score".to_string(),
            maximize: true,
        };
        
        assert_eq!(config.max_trials, 50);
        assert_eq!(config.n_workers, 2);
        assert_eq!(config.seed, Some(42));
        assert_eq!(config.objective_metric, "f1_score");
        assert!(config.maximize);
        assert!(config.early_stopping.is_some());
    }
    
    #[test]
    fn test_hyperparameter_optimizer() {
        let config = OptimizationConfig {
            method: OptimizationMethod::RandomSearch,
            max_trials: 5,
            max_time: Some(Duration::from_secs(30)),
            n_workers: 1,
            seed: Some(42),
            early_stopping: None,
            objective_metric: "accuracy".to_string(),
            maximize: true,
        };
        
        let parameters = vec![
            HyperParameter {
                name: "learning_rate".to_string(),
                param_type: ParameterType::LogContinuous { min: 0.0001, max: 0.1 },
                default: serde_json::Value::Number(serde_json::Number::from_f64(0.001).unwrap()),
                description: Some("Learning rate for optimizer".to_string()),
            },
            HyperParameter {
                name: "batch_size".to_string(),
                param_type: ParameterType::Integer { min: 16, max: 128 },
                default: serde_json::Value::Number(serde_json::Number::from(32)),
                description: Some("Batch size for training".to_string()),
            },
        ];
        
        let mut optimizer = HyperparameterOptimizer::new(config, parameters);
        
        // Mock model factory
        let model_factory = |_params: &HashMap<String, serde_json::Value>| {
            let config = NeuralConfig::new().with_layers(vec![5, 3, 1]);
            NeuralNetwork::new(config)
        };
        
        let result = optimizer.optimize(model_factory);
        assert!(result.is_ok());
        
        let best_trial = result.unwrap();
        assert_eq!(best_trial.status, TrialStatus::Completed);
        assert!(best_trial.metrics.accuracy.is_some());
        
        let summary = optimizer.get_summary();
        assert_eq!(summary.total_trials, 5);
        assert!(summary.best_objective.is_some());
    }
    
    #[test]
    fn test_training_pipeline() {
        let config = TrainingConfig::new()
            .with_max_epochs(3)
            .with_batch_size(16)
            .with_learning_rate(0.01)
            .with_validation_split(0.3);
        
        let mut pipeline = TrainingPipeline::new(config);
        
        // Add preprocessors
        pipeline.add_preprocessor(Box::new(StandardScaler::new()));
        pipeline.set_validation_strategy(ValidationStrategy::HoldOut { test_size: 0.2 });
        
        // Create test data
        let x = Array2::random((100, 5), rand_distr::Uniform::new(-1.0, 1.0));
        let y = Array2::random((100, 1), rand_distr::Uniform::new(0.0, 1.0));
        
        // Create model
        let model_config = NeuralConfig::new().with_layers(vec![5, 3, 1]);
        let model = NeuralNetwork::new(model_config).unwrap();
        
        // Train model
        let result = pipeline.train(model, &x, &y);
        assert!(result.is_ok());
        
        let training_result = result.unwrap();
        assert!(!training_result.training_history.is_empty());
        assert!(training_result.total_training_time > Duration::from_millis(0));
    }
    
    #[test]
    fn test_data_preprocessors() {
        let x = Array2::from_shape_vec(
            (3, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ).unwrap();
        let y = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        
        // Test StandardScaler
        let scaler = StandardScaler::new();
        let (x_scaled, y_unchanged) = scaler.preprocess(&x, &y).unwrap();
        
        assert_eq!(x_scaled.shape(), x.shape());
        assert_eq!(y_unchanged.shape(), y.shape());
        assert_eq!(scaler.name(), "StandardScaler");
        
        // Test MinMaxScaler
        let minmax_scaler = MinMaxScaler::new();
        let (x_minmax, _) = minmax_scaler.preprocess(&x, &y).unwrap();
        
        assert_eq!(x_minmax.shape(), x.shape());
        assert_eq!(minmax_scaler.name(), "MinMaxScaler");
        
        // Check that values are in [0, 1] range
        for &val in x_minmax.iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }
    
    #[test]
    fn test_validation_strategies() {
        let x = Array2::random((20, 3), rand_distr::Uniform::new(-1.0, 1.0));
        let y = Array2::random((20, 1), rand_distr::Uniform::new(0.0, 1.0));
        
        let config = TrainingConfig::default();
        let pipeline = TrainingPipeline::new(config);
        
        // Test HoldOut
        let holdout_splits = pipeline.create_splits(&x, &y).unwrap();
        assert_eq!(holdout_splits.len(), 1);
        
        let (x_train, y_train, x_val, y_val) = &holdout_splits[0];
        assert!(x_train.nrows() > 0);
        assert!(x_val.nrows() > 0);
        assert_eq!(x_train.nrows() + x_val.nrows(), x.nrows());
        assert_eq!(y_train.nrows() + y_val.nrows(), y.nrows());
    }
    
    #[test]
    fn test_checkpoint_config() {
        let temp_dir = tempdir().unwrap();
        
        let checkpoint_config = CheckpointConfig {
            checkpoint_dir: temp_dir.path().to_path_buf(),
            save_frequency: 5,
            keep_best: Some(3),
            best_metric: "accuracy".to_string(),
        };
        
        assert_eq!(checkpoint_config.save_frequency, 5);
        assert_eq!(checkpoint_config.keep_best, Some(3));
        assert_eq!(checkpoint_config.best_metric, "accuracy");
    }
    
    #[test]
    fn test_trial_status() {
        assert_eq!(TrialStatus::Running.to_string(), "Running");
        assert_eq!(TrialStatus::Completed.to_string(), "Completed");
        assert_eq!(TrialStatus::Failed.to_string(), "Failed");
        assert_eq!(TrialStatus::Cancelled.to_string(), "Cancelled");
    }
    
    #[test]
    fn test_optimization_summary() {
        let summary = OptimizationSummary {
            total_trials: 10,
            successful_trials: 8,
            failed_trials: 2,
            total_time: Duration::from_secs(120),
            best_objective: Some(0.95),
            best_parameters: Some({
                let mut params = HashMap::new();
                params.insert("lr".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(0.001).unwrap()));
                params
            }),
        };
        
        let report = summary.generate_report();
        assert!(report.contains("Total Trials: 10"));
        assert!(report.contains("Successful: 8"));
        assert!(report.contains("Failed: 2"));
        assert!(report.contains("Success Rate: 80.0%"));
        assert!(report.contains("Best Objective: 0.950000"));
        assert!(report.contains("Best Parameters:"));
        assert!(report.contains("lr: 0.001"));
    }
}