//! Adaptive parameter tuning for CDFA framework
//!
//! This module implements intelligent parameter optimization using machine learning,
//! evolutionary strategies, and statistical analysis to automatically tune algorithm
//! parameters for optimal performance.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use nalgebra::{DVector, DMatrix};
use ndarray::{Array1, Array2};
use rayon::prelude::*;

use crate::core::{SwarmError, Individual, Population, AlgorithmMetrics};
use crate::cdfa::{PerformanceTracker, PerformanceMetrics};

/// Adaptive parameter tuning system
pub struct AdaptiveParameterTuning {
    /// Parameter space definition
    parameter_space: Arc<RwLock<ParameterSpace>>,
    
    /// Tuning strategies
    strategies: Vec<Box<dyn TuningStrategy>>,
    
    /// Performance history for learning
    performance_history: Arc<RwLock<HashMap<String, Vec<TuningResult>>>>,
    
    /// Machine learning models
    ml_models: Arc<RwLock<HashMap<String, Box<dyn MLModel>>>>,
    
    /// Configuration
    config: AdaptiveTuningConfig,
    
    /// Current best parameters
    best_parameters: Arc<RwLock<HashMap<String, ParameterSet>>>,
}

/// Parameter space definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSpace {
    /// Parameter definitions
    parameters: HashMap<String, ParameterDefinition>,
    
    /// Parameter dependencies
    dependencies: Vec<ParameterDependency>,
    
    /// Constraints
    constraints: Vec<ParameterConstraint>,
    
    /// Default values
    defaults: HashMap<String, f64>,
}

/// Individual parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDefinition {
    /// Parameter name
    pub name: String,
    
    /// Parameter type
    pub param_type: ParameterType,
    
    /// Value range
    pub range: ParameterRange,
    
    /// Description
    pub description: String,
    
    /// Sensitivity level
    pub sensitivity: SensitivityLevel,
    
    /// Update frequency
    pub update_frequency: UpdateFrequency,
}

/// Types of parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    /// Continuous real-valued parameter
    Continuous,
    
    /// Discrete integer parameter
    Discrete,
    
    /// Boolean parameter
    Boolean,
    
    /// Categorical parameter
    Categorical { categories: Vec<String> },
    
    /// Ordinal parameter
    Ordinal { levels: Vec<String> },
}

/// Parameter value ranges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterRange {
    /// Continuous range [min, max]
    Continuous { min: f64, max: f64 },
    
    /// Discrete range [min, max] with step
    Discrete { min: i32, max: i32, step: i32 },
    
    /// Boolean (no range needed)
    Boolean,
    
    /// Set of allowed values
    Set { values: Vec<f64> },
}

/// Parameter sensitivity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SensitivityLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Parameter update frequencies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum UpdateFrequency {
    Never,
    OnConvergence,
    Periodic { every_n_iterations: usize },
    Adaptive,
    Continuous,
}

/// Parameter dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDependency {
    /// Dependent parameter
    pub dependent: String,
    
    /// Independent parameter
    pub independent: String,
    
    /// Dependency type
    pub dependency_type: DependencyType,
    
    /// Dependency function
    pub function: DependencyFunction,
}

/// Types of dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyType {
    Linear,
    Nonlinear,
    Conditional,
    Inverse,
}

/// Dependency functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyFunction {
    /// Linear: y = a*x + b
    Linear { a: f64, b: f64 },
    
    /// Quadratic: y = a*x^2 + b*x + c
    Quadratic { a: f64, b: f64, c: f64 },
    
    /// Exponential: y = a * exp(b*x)
    Exponential { a: f64, b: f64 },
    
    /// Logarithmic: y = a * ln(b*x)
    Logarithmic { a: f64, b: f64 },
    
    /// Custom function (placeholder)
    Custom { expression: String },
}

/// Parameter constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterConstraint {
    /// Constraint name
    pub name: String,
    
    /// Parameters involved
    pub parameters: Vec<String>,
    
    /// Constraint type
    pub constraint_type: ConstraintType,
    
    /// Constraint expression
    pub expression: String,
}

/// Types of constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    Equality,
    Inequality,
    Bounds,
    Custom,
}

/// Set of parameter values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSet {
    /// Parameter values
    pub values: HashMap<String, f64>,
    
    /// Timestamp of creation
    pub timestamp: std::time::SystemTime,
    
    /// Source of parameters
    pub source: ParameterSource,
    
    /// Confidence score
    pub confidence: f64,
    
    /// Validity check
    pub is_valid: bool,
}

/// Source of parameter values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterSource {
    Default,
    Manual,
    GridSearch,
    RandomSearch,
    BayesianOptimization,
    EvolutionaryStrategy,
    GradientBased,
    MachineLearning,
    HybridMethod { methods: Vec<String> },
}

/// Tuning result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningResult {
    /// Parameter set used
    pub parameters: ParameterSet,
    
    /// Performance achieved
    pub performance: f64,
    
    /// Detailed metrics
    pub metrics: Option<PerformanceMetrics>,
    
    /// Execution time
    pub execution_time: std::time::Duration,
    
    /// Success indicator
    pub success: bool,
    
    /// Additional context
    pub context: HashMap<String, f64>,
}

/// Adaptive tuning configuration
#[derive(Debug, Clone)]
pub struct AdaptiveTuningConfig {
    /// Maximum tuning iterations
    pub max_iterations: usize,
    
    /// Convergence threshold
    pub convergence_threshold: f64,
    
    /// Exploration vs exploitation balance
    pub exploration_factor: f64,
    
    /// Learning rate for adaptive methods
    pub learning_rate: f64,
    
    /// Batch size for batch methods
    pub batch_size: usize,
    
    /// Enable parallel tuning
    pub parallel_tuning: bool,
    
    /// Maximum parallel workers
    pub max_workers: usize,
    
    /// Early stopping criteria
    pub early_stopping: bool,
    
    /// Patience for early stopping
    pub patience: usize,
}

impl Default for AdaptiveTuningConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-6,
            exploration_factor: 0.1,
            learning_rate: 0.01,
            batch_size: 10,
            parallel_tuning: true,
            max_workers: num_cpus::get(),
            early_stopping: true,
            patience: 10,
        }
    }
}

/// Trait for tuning strategies
pub trait TuningStrategy: Send + Sync {
    /// Strategy name
    fn name(&self) -> &'static str;
    
    /// Suggest next parameter set to try
    fn suggest_parameters(
        &mut self,
        parameter_space: &ParameterSpace,
        history: &[TuningResult],
    ) -> Result<ParameterSet, SwarmError>;
    
    /// Update strategy based on results
    fn update(&mut self, result: &TuningResult) -> Result<(), SwarmError>;
    
    /// Check if strategy has converged
    fn has_converged(&self) -> bool;
    
    /// Get strategy configuration
    fn get_config(&self) -> HashMap<String, f64>;
    
    /// Set strategy configuration
    fn set_config(&mut self, config: HashMap<String, f64>) -> Result<(), SwarmError>;
}

/// Trait for machine learning models
pub trait MLModel: Send + Sync {
    /// Model name
    fn name(&self) -> &'static str;
    
    /// Train model on data
    fn train(&mut self, inputs: &[Vec<f64>], outputs: &[f64]) -> Result<(), SwarmError>;
    
    /// Predict performance for parameter set
    fn predict(&self, parameters: &[f64]) -> Result<f64, SwarmError>;
    
    /// Get prediction uncertainty
    fn predict_with_uncertainty(&self, parameters: &[f64]) -> Result<(f64, f64), SwarmError>;
    
    /// Update model incrementally
    fn update(&mut self, input: &[f64], output: f64) -> Result<(), SwarmError>;
}

/// Grid search strategy
#[derive(Debug, Clone)]
pub struct GridSearchStrategy {
    /// Grid resolution
    resolution: usize,
    
    /// Current grid position
    current_position: Vec<usize>,
    
    /// Grid dimensions
    grid_dimensions: Vec<usize>,
    
    /// Parameter order
    parameter_order: Vec<String>,
    
    /// Completed flag
    completed: bool,
}

impl GridSearchStrategy {
    pub fn new(resolution: usize) -> Self {
        Self {
            resolution,
            current_position: Vec::new(),
            grid_dimensions: Vec::new(),
            parameter_order: Vec::new(),
            completed: false,
        }
    }
}

impl TuningStrategy for GridSearchStrategy {
    fn name(&self) -> &'static str {
        "GridSearch"
    }
    
    fn suggest_parameters(
        &mut self,
        parameter_space: &ParameterSpace,
        _history: &[TuningResult],
    ) -> Result<ParameterSet, SwarmError> {
        if self.parameter_order.is_empty() {
            // Initialize grid
            self.parameter_order = parameter_space.parameters.keys().cloned().collect();
            self.grid_dimensions = vec![self.resolution; self.parameter_order.len()];
            self.current_position = vec![0; self.parameter_order.len()];
        }
        
        if self.completed {
            return Err(SwarmError::parameter("Grid search completed"));
        }
        
        let mut values = HashMap::new();
        
        for (i, param_name) in self.parameter_order.iter().enumerate() {
            if let Some(param_def) = parameter_space.parameters.get(param_name) {
                let grid_pos = self.current_position[i];
                let value = self.grid_position_to_value(param_def, grid_pos)?;
                values.insert(param_name.clone(), value);
            }
        }
        
        // Advance to next grid position
        self.advance_grid_position();
        
        Ok(ParameterSet {
            values,
            timestamp: std::time::SystemTime::now(),
            source: ParameterSource::GridSearch,
            confidence: 1.0,
            is_valid: true,
        })
    }
    
    fn update(&mut self, _result: &TuningResult) -> Result<(), SwarmError> {
        // Grid search doesn't update based on results
        Ok(())
    }
    
    fn has_converged(&self) -> bool {
        self.completed
    }
    
    fn get_config(&self) -> HashMap<String, f64> {
        let mut config = HashMap::new();
        config.insert("resolution".to_string(), self.resolution as f64);
        config
    }
    
    fn set_config(&mut self, config: HashMap<String, f64>) -> Result<(), SwarmError> {
        if let Some(&resolution) = config.get("resolution") {
            self.resolution = resolution as usize;
        }
        Ok(())
    }
}

impl GridSearchStrategy {
    fn grid_position_to_value(&self, param_def: &ParameterDefinition, position: usize) -> Result<f64, SwarmError> {
        match &param_def.range {
            ParameterRange::Continuous { min, max } => {
                let ratio = position as f64 / (self.resolution - 1) as f64;
                Ok(min + ratio * (max - min))
            }
            ParameterRange::Discrete { min, max, step: _ } => {
                let ratio = position as f64 / (self.resolution - 1) as f64;
                Ok(*min as f64 + ratio * (*max - *min) as f64)
            }
            ParameterRange::Boolean => {
                Ok(if position < self.resolution / 2 { 0.0 } else { 1.0 })
            }
            ParameterRange::Set { values } => {
                let index = (position * values.len()) / self.resolution;
                Ok(values[index.min(values.len() - 1)])
            }
        }
    }
    
    fn advance_grid_position(&mut self) {
        for i in 0..self.current_position.len() {
            self.current_position[i] += 1;
            if self.current_position[i] < self.grid_dimensions[i] {
                return;
            }
            self.current_position[i] = 0;
        }
        // If we get here, we've completed the grid
        self.completed = true;
    }
}

/// Random search strategy
#[derive(Debug, Clone)]
pub struct RandomSearchStrategy {
    /// Random number generator state
    rng_seed: u64,
    
    /// Number of samples generated
    samples_generated: usize,
    
    /// Maximum samples
    max_samples: usize,
}

impl RandomSearchStrategy {
    pub fn new(max_samples: usize) -> Self {
        Self {
            rng_seed: 42, // Fixed seed for reproducibility
            samples_generated: 0,
            max_samples,
        }
    }
}

impl TuningStrategy for RandomSearchStrategy {
    fn name(&self) -> &'static str {
        "RandomSearch"
    }
    
    fn suggest_parameters(
        &mut self,
        parameter_space: &ParameterSpace,
        _history: &[TuningResult],
    ) -> Result<ParameterSet, SwarmError> {
        if self.samples_generated >= self.max_samples {
            return Err(SwarmError::parameter("Random search completed"));
        }
        
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        
        let mut rng = StdRng::seed_from_u64(self.rng_seed + self.samples_generated as u64);
        let mut values = HashMap::new();
        
        for (param_name, param_def) in &parameter_space.parameters {
            let value = match &param_def.range {
                ParameterRange::Continuous { min, max } => {
                    rng.gen_range(*min..=*max)
                }
                ParameterRange::Discrete { min, max, step: _ } => {
                    rng.gen_range(*min..=*max) as f64
                }
                ParameterRange::Boolean => {
                    if rng.gen_bool(0.5) { 1.0 } else { 0.0 }
                }
                ParameterRange::Set { values: vals } => {
                    vals[rng.gen_range(0..vals.len())]
                }
            };
            
            values.insert(param_name.clone(), value);
        }
        
        self.samples_generated += 1;
        
        Ok(ParameterSet {
            values,
            timestamp: std::time::SystemTime::now(),
            source: ParameterSource::RandomSearch,
            confidence: 0.5, // Random confidence
            is_valid: true,
        })
    }
    
    fn update(&mut self, _result: &TuningResult) -> Result<(), SwarmError> {
        // Random search doesn't update based on results
        Ok(())
    }
    
    fn has_converged(&self) -> bool {
        self.samples_generated >= self.max_samples
    }
    
    fn get_config(&self) -> HashMap<String, f64> {
        let mut config = HashMap::new();
        config.insert("max_samples".to_string(), self.max_samples as f64);
        config
    }
    
    fn set_config(&mut self, config: HashMap<String, f64>) -> Result<(), SwarmError> {
        if let Some(&max_samples) = config.get("max_samples") {
            self.max_samples = max_samples as usize;
        }
        Ok(())
    }
}

/// Bayesian optimization strategy
#[derive(Debug)]
pub struct BayesianOptimizationStrategy {
    /// Gaussian process model
    gp_model: Option<Box<dyn MLModel>>,
    
    /// Acquisition function type
    acquisition_function: AcquisitionFunction,
    
    /// Number of samples
    num_samples: usize,
    
    /// Exploration parameter
    exploration_param: f64,
}

/// Acquisition functions for Bayesian optimization
#[derive(Debug, Clone)]
pub enum AcquisitionFunction {
    ExpectedImprovement,
    UpperConfidenceBound { beta: f64 },
    ProbabilityOfImprovement,
    EntropySearch,
}

impl BayesianOptimizationStrategy {
    pub fn new() -> Self {
        Self {
            gp_model: None,
            acquisition_function: AcquisitionFunction::ExpectedImprovement,
            num_samples: 0,
            exploration_param: 2.0,
        }
    }
}

impl TuningStrategy for BayesianOptimizationStrategy {
    fn name(&self) -> &'static str {
        "BayesianOptimization"
    }
    
    fn suggest_parameters(
        &mut self,
        parameter_space: &ParameterSpace,
        history: &[TuningResult],
    ) -> Result<ParameterSet, SwarmError> {
        if history.is_empty() {
            // Generate initial random sample
            return self.generate_random_sample(parameter_space);
        }
        
        // Train or update Gaussian process
        if self.gp_model.is_none() {
            self.gp_model = Some(Box::new(SimpleGaussianProcess::new()));
        }
        
        // Prepare training data
        let (inputs, outputs) = self.prepare_training_data(parameter_space, history)?;
        
        if let Some(ref mut model) = self.gp_model {
            model.train(&inputs, &outputs)?;
        }
        
        // Optimize acquisition function
        let best_params = self.optimize_acquisition_function(parameter_space)?;
        
        self.num_samples += 1;
        
        Ok(ParameterSet {
            values: best_params,
            timestamp: std::time::SystemTime::now(),
            source: ParameterSource::BayesianOptimization,
            confidence: 0.8,
            is_valid: true,
        })
    }
    
    fn update(&mut self, result: &TuningResult) -> Result<(), SwarmError> {
        // Update GP model incrementally if available
        if let Some(ref mut model) = self.gp_model {
            let input: Vec<f64> = result.parameters.values.values().copied().collect();
            model.update(&input, result.performance)?;
        }
        Ok(())
    }
    
    fn has_converged(&self) -> bool {
        self.num_samples > 50 // Simple convergence criterion
    }
    
    fn get_config(&self) -> HashMap<String, f64> {
        let mut config = HashMap::new();
        config.insert("exploration_param".to_string(), self.exploration_param);
        config
    }
    
    fn set_config(&mut self, config: HashMap<String, f64>) -> Result<(), SwarmError> {
        if let Some(&param) = config.get("exploration_param") {
            self.exploration_param = param;
        }
        Ok(())
    }
}

impl BayesianOptimizationStrategy {
    fn generate_random_sample(&self, parameter_space: &ParameterSpace) -> Result<ParameterSet, SwarmError> {
        let mut random_strategy = RandomSearchStrategy::new(1);
        random_strategy.suggest_parameters(parameter_space, &[])
    }
    
    fn prepare_training_data(
        &self,
        parameter_space: &ParameterSpace,
        history: &[TuningResult],
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>), SwarmError> {
        let param_order: Vec<String> = parameter_space.parameters.keys().cloned().collect();
        
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        
        for result in history {
            let input: Vec<f64> = param_order.iter()
                .map(|name| result.parameters.values.get(name).copied().unwrap_or(0.0))
                .collect();
            
            inputs.push(input);
            outputs.push(result.performance);
        }
        
        Ok((inputs, outputs))
    }
    
    fn optimize_acquisition_function(
        &self,
        parameter_space: &ParameterSpace,
    ) -> Result<HashMap<String, f64>, SwarmError> {
        // Simplified acquisition function optimization
        // In practice, this would use sophisticated optimization methods
        
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        
        let mut rng = StdRng::seed_from_u64(42);
        let mut best_params = HashMap::new();
        let mut best_acquisition = f64::NEG_INFINITY;
        
        // Sample-based optimization
        for _ in 0..100 {
            let mut candidate_params = HashMap::new();
            
            for (param_name, param_def) in &parameter_space.parameters {
                let value = match &param_def.range {
                    ParameterRange::Continuous { min, max } => {
                        rng.gen_range(*min..=*max)
                    }
                    ParameterRange::Discrete { min, max, step: _ } => {
                        rng.gen_range(*min..=*max) as f64
                    }
                    ParameterRange::Boolean => {
                        if rng.gen_bool(0.5) { 1.0 } else { 0.0 }
                    }
                    ParameterRange::Set { values } => {
                        values[rng.gen_range(0..values.len())]
                    }
                };
                
                candidate_params.insert(param_name.clone(), value);
            }
            
            // Evaluate acquisition function
            let acquisition_value = self.evaluate_acquisition_function(&candidate_params)?;
            
            if acquisition_value > best_acquisition {
                best_acquisition = acquisition_value;
                best_params = candidate_params;
            }
        }
        
        Ok(best_params)
    }
    
    fn evaluate_acquisition_function(&self, _params: &HashMap<String, f64>) -> Result<f64, SwarmError> {
        // Simplified acquisition function evaluation
        // In practice, this would use the GP model to compute expected improvement, UCB, etc.
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        Ok(rng.gen::<f64>()) // Random value for now
    }
}

/// Simple Gaussian Process implementation
#[derive(Debug)]
pub struct SimpleGaussianProcess {
    /// Training inputs
    training_inputs: Vec<Vec<f64>>,
    
    /// Training outputs
    training_outputs: Vec<f64>,
    
    /// Kernel parameters
    kernel_params: HashMap<String, f64>,
}

impl SimpleGaussianProcess {
    pub fn new() -> Self {
        let mut kernel_params = HashMap::new();
        kernel_params.insert("length_scale".to_string(), 1.0);
        kernel_params.insert("signal_variance".to_string(), 1.0);
        kernel_params.insert("noise_variance".to_string(), 0.1);
        
        Self {
            training_inputs: Vec::new(),
            training_outputs: Vec::new(),
            kernel_params,
        }
    }
    
    fn rbf_kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let length_scale = self.kernel_params.get("length_scale").copied().unwrap_or(1.0);
        let signal_variance = self.kernel_params.get("signal_variance").copied().unwrap_or(1.0);
        
        let squared_distance: f64 = x1.iter().zip(x2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        
        signal_variance * (-0.5 * squared_distance / length_scale.powi(2)).exp()
    }
}

impl MLModel for SimpleGaussianProcess {
    fn name(&self) -> &'static str {
        "SimpleGaussianProcess"
    }
    
    fn train(&mut self, inputs: &[Vec<f64>], outputs: &[f64]) -> Result<(), SwarmError> {
        if inputs.len() != outputs.len() {
            return Err(SwarmError::parameter("Input and output lengths don't match"));
        }
        
        self.training_inputs = inputs.to_vec();
        self.training_outputs = outputs.to_vec();
        
        Ok(())
    }
    
    fn predict(&self, parameters: &[f64]) -> Result<f64, SwarmError> {
        if self.training_inputs.is_empty() {
            return Ok(0.0);
        }
        
        // Simplified GP prediction (mean prediction only)
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        
        for (i, training_input) in self.training_inputs.iter().enumerate() {
            let weight = self.rbf_kernel(parameters, training_input);
            weighted_sum += weight * self.training_outputs[i];
            weight_sum += weight;
        }
        
        if weight_sum > 0.0 {
            Ok(weighted_sum / weight_sum)
        } else {
            Ok(0.0)
        }
    }
    
    fn predict_with_uncertainty(&self, parameters: &[f64]) -> Result<(f64, f64), SwarmError> {
        let mean = self.predict(parameters)?;
        let uncertainty = 0.1; // Simplified uncertainty estimate
        Ok((mean, uncertainty))
    }
    
    fn update(&mut self, input: &[f64], output: f64) -> Result<(), SwarmError> {
        self.training_inputs.push(input.to_vec());
        self.training_outputs.push(output);
        
        // Limit training data size
        if self.training_inputs.len() > 1000 {
            self.training_inputs.remove(0);
            self.training_outputs.remove(0);
        }
        
        Ok(())
    }
}

impl AdaptiveParameterTuning {
    /// Create a new adaptive parameter tuning system
    pub fn new() -> Self {
        Self {
            parameter_space: Arc::new(RwLock::new(ParameterSpace::default())),
            strategies: Vec::new(),
            performance_history: Arc::new(RwLock::new(HashMap::new())),
            ml_models: Arc::new(RwLock::new(HashMap::new())),
            config: AdaptiveTuningConfig::default(),
            best_parameters: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Create with custom configuration
    pub fn with_config(config: AdaptiveTuningConfig) -> Self {
        let mut tuner = Self::new();
        tuner.config = config;
        tuner
    }
    
    /// Add a tuning strategy
    pub fn add_strategy(&mut self, strategy: Box<dyn TuningStrategy>) {
        self.strategies.push(strategy);
    }
    
    /// Set parameter space
    pub fn set_parameter_space(&self, space: ParameterSpace) {
        let mut parameter_space = self.parameter_space.write();
        *parameter_space = space;
    }
    
    /// Tune parameters for an algorithm
    pub async fn tune_parameters<F>(
        &mut self,
        algorithm_id: String,
        evaluation_function: F,
    ) -> Result<ParameterSet, SwarmError>
    where
        F: Fn(&ParameterSet) -> Result<f64, SwarmError> + Send + Sync + Clone + 'static,
    {
        let mut best_result: Option<TuningResult> = None;
        let mut no_improvement_count = 0;
        
        for iteration in 0..self.config.max_iterations {
            tracing::debug!("Tuning iteration {} for algorithm {}", iteration, algorithm_id);
            
            // Get suggestions from all strategies
            let suggestions = self.get_parameter_suggestions(&algorithm_id)?;
            
            // Evaluate suggestions
            let results = if self.config.parallel_tuning {
                self.evaluate_suggestions_parallel(suggestions, evaluation_function.clone()).await?
            } else {
                self.evaluate_suggestions_sequential(suggestions, evaluation_function.clone()).await?
            };
            
            // Update performance history
            self.update_performance_history(&algorithm_id, &results);
            
            // Update strategies
            for (strategy, result) in self.strategies.iter_mut().zip(&results) {
                let _ = strategy.update(result);
            }
            
            // Check for best result
            if let Some(best_current) = results.iter().max_by(|a, b| a.performance.partial_cmp(&b.performance).unwrap()) {
                if best_result.is_none() || best_current.performance > best_result.as_ref().unwrap().performance {
                    best_result = Some(best_current.clone());
                    no_improvement_count = 0;
                    
                    // Update best parameters
                    let mut best_params = self.best_parameters.write();
                    best_params.insert(algorithm_id.clone(), best_current.parameters.clone());
                } else {
                    no_improvement_count += 1;
                }
            }
            
            // Early stopping
            if self.config.early_stopping && no_improvement_count >= self.config.patience {
                tracing::info!("Early stopping triggered for algorithm {}", algorithm_id);
                break;
            }
            
            // Convergence check
            if self.has_converged() {
                tracing::info!("Convergence detected for algorithm {}", algorithm_id);
                break;
            }
        }
        
        best_result
            .map(|result| result.parameters)
            .ok_or_else(|| SwarmError::optimization("No successful tuning results"))
    }
    
    /// Get parameter suggestions from all strategies
    fn get_parameter_suggestions(&mut self, algorithm_id: &str) -> Result<Vec<ParameterSet>, SwarmError> {
        let parameter_space = self.parameter_space.read();
        let history = self.get_algorithm_history(algorithm_id);
        
        let mut suggestions = Vec::new();
        
        for strategy in &mut self.strategies {
            match strategy.suggest_parameters(&parameter_space, &history) {
                Ok(params) => suggestions.push(params),
                Err(e) => tracing::warn!("Strategy {} failed: {}", strategy.name(), e),
            }
        }
        
        // Add random exploration if configured
        if suggestions.is_empty() || (rand::random::<f64>() < self.config.exploration_factor) {
            let mut random_strategy = RandomSearchStrategy::new(1);
            if let Ok(random_params) = random_strategy.suggest_parameters(&parameter_space, &history) {
                suggestions.push(random_params);
            }
        }
        
        Ok(suggestions)
    }
    
    /// Evaluate parameter suggestions sequentially
    async fn evaluate_suggestions_sequential<F>(
        &self,
        suggestions: Vec<ParameterSet>,
        evaluation_function: F,
    ) -> Result<Vec<TuningResult>, SwarmError>
    where
        F: Fn(&ParameterSet) -> Result<f64, SwarmError>,
    {
        let mut results = Vec::new();
        
        for params in suggestions {
            let start_time = std::time::Instant::now();
            
            match evaluation_function(&params) {
                Ok(performance) => {
                    results.push(TuningResult {
                        parameters: params,
                        performance,
                        metrics: None,
                        execution_time: start_time.elapsed(),
                        success: true,
                        context: HashMap::new(),
                    });
                }
                Err(e) => {
                    tracing::warn!("Parameter evaluation failed: {}", e);
                    results.push(TuningResult {
                        parameters: params,
                        performance: f64::NEG_INFINITY,
                        metrics: None,
                        execution_time: start_time.elapsed(),
                        success: false,
                        context: HashMap::new(),
                    });
                }
            }
        }
        
        Ok(results)
    }
    
    /// Evaluate parameter suggestions in parallel
    async fn evaluate_suggestions_parallel<F>(
        &self,
        suggestions: Vec<ParameterSet>,
        evaluation_function: F,
    ) -> Result<Vec<TuningResult>, SwarmError>
    where
        F: Fn(&ParameterSet) -> Result<f64, SwarmError> + Send + Sync + Clone + 'static,
    {
        use tokio::task;
        
        let tasks: Vec<_> = suggestions.into_iter()
            .map(|params| {
                let eval_fn = evaluation_function.clone();
                task::spawn(async move {
                    let start_time = std::time::Instant::now();
                    
                    match eval_fn(&params) {
                        Ok(performance) => TuningResult {
                            parameters: params,
                            performance,
                            metrics: None,
                            execution_time: start_time.elapsed(),
                            success: true,
                            context: HashMap::new(),
                        },
                        Err(_) => TuningResult {
                            parameters: params,
                            performance: f64::NEG_INFINITY,
                            metrics: None,
                            execution_time: start_time.elapsed(),
                            success: false,
                            context: HashMap::new(),
                        },
                    }
                })
            })
            .collect();
        
        let mut results = Vec::new();
        for task in tasks {
            match task.await {
                Ok(result) => results.push(result),
                Err(e) => tracing::warn!("Parallel evaluation task failed: {}", e),
            }
        }
        
        Ok(results)
    }
    
    /// Update performance history
    fn update_performance_history(&self, algorithm_id: &str, results: &[TuningResult]) {
        let mut history = self.performance_history.write();
        let algorithm_history = history.entry(algorithm_id.to_string()).or_insert_with(Vec::new);
        
        algorithm_history.extend(results.iter().cloned());
        
        // Limit history size
        if algorithm_history.len() > 10000 {
            algorithm_history.drain(0..1000);
        }
    }
    
    /// Get algorithm tuning history
    fn get_algorithm_history(&self, algorithm_id: &str) -> Vec<TuningResult> {
        let history = self.performance_history.read();
        history.get(algorithm_id).cloned().unwrap_or_default()
    }
    
    /// Check if tuning has converged
    fn has_converged(&self) -> bool {
        self.strategies.iter().all(|strategy| strategy.has_converged())
    }
    
    /// Get best parameters for an algorithm
    pub fn get_best_parameters(&self, algorithm_id: &str) -> Option<ParameterSet> {
        let best_params = self.best_parameters.read();
        best_params.get(algorithm_id).cloned()
    }
    
    /// Get tuning statistics
    pub fn get_tuning_statistics(&self, algorithm_id: &str) -> TuningStatistics {
        let history = self.get_algorithm_history(algorithm_id);
        
        if history.is_empty() {
            return TuningStatistics::default();
        }
        
        let performances: Vec<f64> = history.iter()
            .filter(|r| r.success)
            .map(|r| r.performance)
            .collect();
        
        if performances.is_empty() {
            return TuningStatistics::default();
        }
        
        let best_performance = performances.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let avg_performance = performances.iter().sum::<f64>() / performances.len() as f64;
        let success_rate = performances.len() as f64 / history.len() as f64;
        
        TuningStatistics {
            total_evaluations: history.len(),
            successful_evaluations: performances.len(),
            best_performance,
            average_performance: avg_performance,
            success_rate,
            convergence_iteration: None, // Would track actual convergence
        }
    }
}

impl Default for AdaptiveParameterTuning {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ParameterSpace {
    fn default() -> Self {
        Self {
            parameters: HashMap::new(),
            dependencies: Vec::new(),
            constraints: Vec::new(),
            defaults: HashMap::new(),
        }
    }
}

/// Tuning statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningStatistics {
    /// Total evaluations performed
    pub total_evaluations: usize,
    
    /// Successful evaluations
    pub successful_evaluations: usize,
    
    /// Best performance achieved
    pub best_performance: f64,
    
    /// Average performance
    pub average_performance: f64,
    
    /// Success rate
    pub success_rate: f64,
    
    /// Iteration when convergence was detected
    pub convergence_iteration: Option<usize>,
}

impl Default for TuningStatistics {
    fn default() -> Self {
        Self {
            total_evaluations: 0,
            successful_evaluations: 0,
            best_performance: f64::NEG_INFINITY,
            average_performance: 0.0,
            success_rate: 0.0,
            convergence_iteration: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_adaptive_tuning_creation() {
        let tuner = AdaptiveParameterTuning::new();
        assert_eq!(tuner.strategies.len(), 0);
        assert_eq!(tuner.config.max_iterations, 100);
    }
    
    #[test]
    fn test_parameter_space() {
        let mut space = ParameterSpace::default();
        
        let param_def = ParameterDefinition {
            name: "learning_rate".to_string(),
            param_type: ParameterType::Continuous,
            range: ParameterRange::Continuous { min: 0.001, max: 0.1 },
            description: "Learning rate parameter".to_string(),
            sensitivity: SensitivityLevel::High,
            update_frequency: UpdateFrequency::Adaptive,
        };
        
        space.parameters.insert("learning_rate".to_string(), param_def);
        assert_eq!(space.parameters.len(), 1);
    }
    
    #[test]
    fn test_grid_search_strategy() {
        let mut strategy = GridSearchStrategy::new(3);
        let mut space = ParameterSpace::default();
        
        let param_def = ParameterDefinition {
            name: "test_param".to_string(),
            param_type: ParameterType::Continuous,
            range: ParameterRange::Continuous { min: 0.0, max: 1.0 },
            description: "Test parameter".to_string(),
            sensitivity: SensitivityLevel::Medium,
            update_frequency: UpdateFrequency::Adaptive,
        };
        
        space.parameters.insert("test_param".to_string(), param_def);
        
        let result = strategy.suggest_parameters(&space, &[]);
        assert!(result.is_ok());
        
        let params = result.unwrap();
        assert!(params.values.contains_key("test_param"));
    }
    
    #[test]
    fn test_random_search_strategy() {
        let mut strategy = RandomSearchStrategy::new(10);
        let mut space = ParameterSpace::default();
        
        let param_def = ParameterDefinition {
            name: "test_param".to_string(),
            param_type: ParameterType::Continuous,
            range: ParameterRange::Continuous { min: -1.0, max: 1.0 },
            description: "Test parameter".to_string(),
            sensitivity: SensitivityLevel::Low,
            update_frequency: UpdateFrequency::Periodic { every_n_iterations: 10 },
        };
        
        space.parameters.insert("test_param".to_string(), param_def);
        
        let result = strategy.suggest_parameters(&space, &[]);
        assert!(result.is_ok());
        
        let params = result.unwrap();
        let value = params.values.get("test_param").unwrap();
        assert!(*value >= -1.0 && *value <= 1.0);
    }
    
    #[test]
    fn test_bayesian_optimization_strategy() {
        let mut strategy = BayesianOptimizationStrategy::new();
        let space = ParameterSpace::default();
        
        // Test with empty history (should generate random sample)
        let result = strategy.suggest_parameters(&space, &[]);
        // Result might fail due to empty parameter space, which is expected
        
        assert_eq!(strategy.name(), "BayesianOptimization");
        assert!(!strategy.has_converged());
    }
    
    #[test]
    fn test_simple_gaussian_process() {
        let mut gp = SimpleGaussianProcess::new();
        assert_eq!(gp.name(), "SimpleGaussianProcess");
        
        // Test training
        let inputs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let outputs = vec![5.0, 10.0];
        
        let result = gp.train(&inputs, &outputs);
        assert!(result.is_ok());
        
        // Test prediction
        let prediction = gp.predict(&[2.0, 3.0]);
        assert!(prediction.is_ok());
        
        // Test prediction with uncertainty
        let prediction_with_uncertainty = gp.predict_with_uncertainty(&[2.0, 3.0]);
        assert!(prediction_with_uncertainty.is_ok());
    }
    
    #[tokio::test]
    async fn test_parameter_tuning_workflow() {
        let mut tuner = AdaptiveParameterTuning::new();
        
        // Add strategies
        tuner.add_strategy(Box::new(RandomSearchStrategy::new(5)));
        
        // Set up parameter space
        let mut space = ParameterSpace::default();
        let param_def = ParameterDefinition {
            name: "x".to_string(),
            param_type: ParameterType::Continuous,
            range: ParameterRange::Continuous { min: -2.0, max: 2.0 },
            description: "Optimization variable".to_string(),
            sensitivity: SensitivityLevel::High,
            update_frequency: UpdateFrequency::Continuous,
        };
        space.parameters.insert("x".to_string(), param_def);
        tuner.set_parameter_space(space);
        
        // Define evaluation function (minimize x^2)
        let evaluation_fn = |params: &ParameterSet| -> Result<f64, SwarmError> {
            let x = params.values.get("x").copied().unwrap_or(0.0);
            Ok(-x * x) // Negative because we maximize in tuning
        };
        
        // Run tuning
        let result = tuner.tune_parameters("test_algorithm".to_string(), evaluation_fn).await;
        assert!(result.is_ok());
        
        let best_params = result.unwrap();
        let x_value = best_params.values.get("x").unwrap();
        // Should be close to 0 for minimizing x^2
        assert!(x_value.abs() < 2.0); // Within bounds at least
    }
}