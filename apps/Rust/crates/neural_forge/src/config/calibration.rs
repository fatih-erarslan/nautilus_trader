//! Calibration configuration for uncertainty quantification

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::error::{Result, NeuralForgeError};

/// Calibration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// Calibration methods to apply
    pub methods: Vec<CalibrationMethod>,
    
    /// Validation split for calibration
    pub calibration_split: f64,
    
    /// Cross-validation for calibration
    pub cross_validation: Option<CalibrationCrossValidation>,
    
    /// Temperature scaling configuration
    pub temperature_scaling: Option<TemperatureScalingConfig>,
    
    /// Platt scaling configuration
    pub platt_scaling: Option<PlattScalingConfig>,
    
    /// Isotonic regression configuration
    pub isotonic_regression: Option<IsotonicRegressionConfig>,
    
    /// Conformal prediction configuration
    pub conformal_prediction: Option<ConformalPredictionConfig>,
    
    /// Bayesian calibration configuration
    pub bayesian_calibration: Option<BayesianCalibrationConfig>,
    
    /// Ensemble calibration configuration
    pub ensemble_calibration: Option<EnsembleCalibrationConfig>,
    
    /// Post-hoc calibration configuration
    pub post_hoc: Option<PostHocCalibrationConfig>,
    
    /// Calibration evaluation metrics
    pub evaluation_metrics: Vec<CalibrationMetric>,
    
    /// Save calibrated model
    pub save_calibrated: bool,
    
    /// Output directory for calibration results
    pub output_dir: std::path::PathBuf,
}

/// Calibration methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CalibrationMethod {
    /// Temperature scaling
    TemperatureScaling,
    
    /// Platt scaling (sigmoid calibration)
    PlattScaling,
    
    /// Isotonic regression
    IsotonicRegression,
    
    /// Conformal prediction
    ConformalPrediction,
    
    /// Bayesian model averaging
    BayesianCalibration,
    
    /// Ensemble calibration
    EnsembleCalibration,
    
    /// Histogram binning
    HistogramBinning { num_bins: usize },
    
    /// Beta calibration
    BetaCalibration,
    
    /// Dirichlet calibration (multi-class)
    DirichletCalibration,
    
    /// Matrix scaling (multi-class)
    MatrixScaling,
    
    /// Custom calibration method
    Custom {
        name: String,
        params: HashMap<String, serde_json::Value>,
    },
}

/// Cross-validation for calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationCrossValidation {
    /// Number of folds
    pub n_folds: u32,
    
    /// Stratified cross-validation
    pub stratified: bool,
    
    /// Random state
    pub random_state: Option<u64>,
}

/// Temperature scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureScalingConfig {
    /// Optimization method
    pub optimizer: TemperatureOptimizer,
    
    /// Maximum iterations
    pub max_iter: u32,
    
    /// Convergence tolerance
    pub tol: f64,
    
    /// Initial temperature
    pub initial_temperature: f64,
    
    /// Temperature bounds
    pub temperature_bounds: (f64, f64),
    
    /// Use validation set for temperature optimization
    pub use_validation: bool,
    
    /// Regularization strength
    pub regularization: Option<f64>,
}

/// Temperature optimization methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemperatureOptimizer {
    /// L-BFGS optimizer
    LBFGS {
        history_size: usize,
        line_search: Option<String>,
    },
    
    /// Nelder-Mead simplex
    NelderMead {
        initial_simplex: Option<Vec<f64>>,
        adaptive: bool,
    },
    
    /// Grid search
    GridSearch {
        temperature_range: (f64, f64),
        num_points: usize,
    },
    
    /// Bayesian optimization
    BayesianOptimization {
        acquisition_function: String,
        num_iter: u32,
    },
    
    /// Gradient descent
    GradientDescent {
        learning_rate: f64,
        momentum: Option<f64>,
    },
}

/// Platt scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlattScalingConfig {
    /// Maximum iterations
    pub max_iter: u32,
    
    /// Convergence tolerance
    pub tol: f64,
    
    /// Regularization strength
    pub reg_param: f64,
    
    /// Prior class counts
    pub prior0: Option<f64>,
    pub prior1: Option<f64>,
}

/// Isotonic regression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsotonicRegressionConfig {
    /// Increasing constraint
    pub increasing: bool,
    
    /// Out-of-bounds handling
    pub out_of_bounds: OutOfBoundsHandling,
    
    /// Interpolation method
    pub interpolation: InterpolationMethod,
}

/// Out-of-bounds handling for isotonic regression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutOfBoundsHandling {
    /// Clip to bounds
    Clip,
    
    /// Not a number
    NaN,
    
    /// Extrapolate
    Extrapolate,
}

/// Interpolation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterpolationMethod {
    Linear,
    Lower,
    Higher,
    Midpoint,
    Nearest,
}

/// Conformal prediction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformalPredictionConfig {
    /// Significance level (alpha)
    pub alpha: f64,
    
    /// Conformal prediction method
    pub method: ConformalMethod,
    
    /// Calibration set size
    pub calibration_size: Option<f64>,
    
    /// Conditional coverage
    pub conditional: Option<ConditionalCoverageConfig>,
    
    /// Adaptive conformal prediction
    pub adaptive: Option<AdaptiveConformalConfig>,
    
    /// Exchangeability test
    pub exchangeability_test: bool,
    
    /// Mondrian conformal prediction
    pub mondrian: Option<MondrianConformalConfig>,
}

/// Conformal prediction methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConformalMethod {
    /// Split conformal prediction
    Split,
    
    /// Cross-conformal prediction
    CrossConformal { n_folds: u32 },
    
    /// Jackknife+ prediction
    JackknifePlus,
    
    /// Conformalized quantile regression
    QuantileRegression { quantiles: Vec<f64> },
    
    /// Locally adaptive conformal prediction
    LocallyAdaptive,
    
    /// Weighted conformal prediction
    Weighted { weight_function: String },
}

/// Conditional coverage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalCoverageConfig {
    /// Conditioning variables
    pub conditioning_vars: Vec<String>,
    
    /// Binning strategy
    pub binning_strategy: BinningStrategy,
    
    /// Minimum samples per bin
    pub min_samples_per_bin: usize,
}

/// Binning strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BinningStrategy {
    /// Equal-width binning
    EqualWidth { num_bins: usize },
    
    /// Equal-frequency binning
    EqualFrequency { num_bins: usize },
    
    /// K-means clustering
    KMeans { k: usize },
    
    /// Custom binning
    Custom { bin_edges: Vec<f64> },
}

/// Adaptive conformal prediction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConformalConfig {
    /// Learning rate for adaptation
    pub learning_rate: f64,
    
    /// Window size for recent observations
    pub window_size: Option<usize>,
    
    /// Forgetting factor
    pub forgetting_factor: Option<f64>,
}

/// Mondrian conformal prediction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MondrianConformalConfig {
    /// Mondrian categories
    pub categories: Vec<String>,
    
    /// Category assignment function
    pub category_function: String,
}

/// Bayesian calibration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianCalibrationConfig {
    /// Prior distribution
    pub prior: PriorDistribution,
    
    /// MCMC configuration
    pub mcmc: Option<MCMCConfig>,
    
    /// Variational inference configuration
    pub variational: Option<VariationalConfig>,
    
    /// Number of posterior samples
    pub num_samples: u32,
    
    /// Burn-in samples
    pub burn_in: u32,
}

/// Prior distributions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PriorDistribution {
    /// Normal prior
    Normal { mean: f64, std: f64 },
    
    /// Uniform prior
    Uniform { low: f64, high: f64 },
    
    /// Gamma prior
    Gamma { alpha: f64, beta: f64 },
    
    /// Beta prior
    Beta { alpha: f64, beta: f64 },
    
    /// Custom prior
    Custom {
        distribution: String,
        params: HashMap<String, f64>,
    },
}

/// MCMC configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCMCConfig {
    /// Sampler type
    pub sampler: MCMCSampler,
    
    /// Number of chains
    pub num_chains: u32,
    
    /// Chain length
    pub chain_length: u32,
    
    /// Thinning factor
    pub thin: u32,
}

/// MCMC samplers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MCMCSampler {
    /// Metropolis-Hastings
    MetropolisHastings { proposal_std: f64 },
    
    /// Hamiltonian Monte Carlo
    HMC {
        step_size: f64,
        num_steps: u32,
    },
    
    /// No-U-Turn Sampler
    NUTS {
        step_size: Option<f64>,
        max_tree_depth: u32,
    },
}

/// Variational inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariationalConfig {
    /// Variational family
    pub family: VariationalFamily,
    
    /// Optimization method
    pub optimizer: String,
    
    /// Number of iterations
    pub num_iter: u32,
    
    /// Learning rate
    pub learning_rate: f64,
}

/// Variational families
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariationalFamily {
    /// Mean-field Gaussian
    MeanFieldGaussian,
    
    /// Full-rank Gaussian
    FullRankGaussian,
    
    /// Normalizing flows
    NormalizingFlows { num_flows: u32 },
}

/// Ensemble calibration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleCalibrationConfig {
    /// Ensemble method
    pub method: EnsembleMethod,
    
    /// Number of models in ensemble
    pub num_models: u32,
    
    /// Bootstrap sampling
    pub bootstrap: bool,
    
    /// Aggregation method
    pub aggregation: AggregationMethod,
    
    /// Diversity promotion
    pub diversity: Option<DiversityConfig>,
}

/// Ensemble methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnsembleMethod {
    /// Bagging
    Bagging,
    
    /// Boosting
    Boosting { learning_rate: f64 },
    
    /// Random subspaces
    RandomSubspaces { feature_fraction: f64 },
    
    /// Dropout ensemble
    DropoutEnsemble { dropout_rate: f64 },
    
    /// Deep ensemble
    DeepEnsemble,
}

/// Aggregation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationMethod {
    /// Simple averaging
    Average,
    
    /// Weighted averaging
    WeightedAverage { weights: Vec<f64> },
    
    /// Median
    Median,
    
    /// Stacking
    Stacking { meta_learner: String },
}

/// Diversity promotion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityConfig {
    /// Diversity metric
    pub metric: DiversityMetric,
    
    /// Diversity weight
    pub weight: f64,
}

/// Diversity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiversityMetric {
    /// Disagreement measure
    Disagreement,
    
    /// Double fault
    DoubleFault,
    
    /// Entropy
    Entropy,
    
    /// Correlation coefficient
    Correlation,
}

/// Post-hoc calibration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostHocCalibrationConfig {
    /// Calibration model
    pub model: PostHocModel,
    
    /// Feature selection
    pub feature_selection: Option<FeatureSelectionConfig>,
    
    /// Cross-validation
    pub cross_validation: bool,
}

/// Post-hoc calibration models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PostHocModel {
    /// Logistic regression
    LogisticRegression { regularization: f64 },
    
    /// Random forest
    RandomForest {
        num_trees: u32,
        max_depth: Option<u32>,
    },
    
    /// Gradient boosting
    GradientBoosting {
        num_rounds: u32,
        learning_rate: f64,
    },
    
    /// Neural network
    NeuralNetwork {
        hidden_sizes: Vec<usize>,
        dropout: f64,
    },
}

/// Feature selection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSelectionConfig {
    /// Selection method
    pub method: FeatureSelectionMethod,
    
    /// Number of features to select
    pub num_features: Option<usize>,
    
    /// Selection threshold
    pub threshold: Option<f64>,
}

/// Feature selection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureSelectionMethod {
    /// Mutual information
    MutualInformation,
    
    /// Chi-square test
    ChiSquare,
    
    /// ANOVA F-test
    ANOVA,
    
    /// Recursive feature elimination
    RFE,
    
    /// L1 regularization
    L1Regularization,
}

/// Calibration evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CalibrationMetric {
    /// Expected Calibration Error
    ECE { num_bins: usize },
    
    /// Maximum Calibration Error
    MCE { num_bins: usize },
    
    /// Overconfidence Error
    OE { num_bins: usize },
    
    /// Underconfidence Error
    UE { num_bins: usize },
    
    /// Adaptive Calibration Error
    ACE,
    
    /// Classwise ECE
    ClasswiseECE { num_bins: usize },
    
    /// Static Calibration Error
    SCE,
    
    /// Thresholded Adaptive Calibration Error
    TACE { threshold: f64 },
    
    /// Brier Score
    BrierScore,
    
    /// Reliability diagram
    ReliabilityDiagram { num_bins: usize },
    
    /// Coverage probability
    Coverage,
    
    /// Average interval width
    AverageWidth,
    
    /// Conditional coverage
    ConditionalCoverage,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            methods: vec![
                CalibrationMethod::TemperatureScaling,
                CalibrationMethod::ConformalPrediction,
            ],
            calibration_split: 0.2,
            cross_validation: None,
            temperature_scaling: Some(TemperatureScalingConfig::default()),
            platt_scaling: None,
            isotonic_regression: None,
            conformal_prediction: Some(ConformalPredictionConfig::default()),
            bayesian_calibration: None,
            ensemble_calibration: None,
            post_hoc: None,
            evaluation_metrics: vec![
                CalibrationMetric::ECE { num_bins: 10 },
                CalibrationMetric::MCE { num_bins: 10 },
                CalibrationMetric::BrierScore,
            ],
            save_calibrated: true,
            output_dir: std::path::PathBuf::from("./calibration_results"),
        }
    }
}

impl Default for TemperatureScalingConfig {
    fn default() -> Self {
        Self {
            optimizer: TemperatureOptimizer::LBFGS {
                history_size: 10,
                line_search: Some("strong_wolfe".to_string()),
            },
            max_iter: 100,
            tol: 1e-6,
            initial_temperature: 1.0,
            temperature_bounds: (0.01, 100.0),
            use_validation: true,
            regularization: None,
        }
    }
}

impl Default for ConformalPredictionConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1, // 90% prediction intervals
            method: ConformalMethod::Split,
            calibration_size: Some(0.2),
            conditional: None,
            adaptive: None,
            exchangeability_test: false,
            mondrian: None,
        }
    }
}

impl CalibrationConfig {
    /// Validate calibration configuration
    pub fn validate(&self) -> Result<()> {
        if self.calibration_split <= 0.0 || self.calibration_split >= 1.0 {
            return Err(NeuralForgeError::calibration(
                "Calibration split must be in (0, 1)"
            ));
        }
        
        // Validate individual method configurations
        if let Some(ref temp_config) = self.temperature_scaling {
            temp_config.validate()?;
        }
        
        if let Some(ref conf_config) = self.conformal_prediction {
            conf_config.validate()?;
        }
        
        Ok(())
    }
    
    /// Create temperature scaling configuration
    pub fn temperature_scaling() -> Self {
        Self {
            methods: vec![CalibrationMethod::TemperatureScaling],
            temperature_scaling: Some(TemperatureScalingConfig::default()),
            ..Default::default()
        }
    }
    
    /// Create conformal prediction configuration
    pub fn conformal_prediction(alpha: f64) -> Self {
        Self {
            methods: vec![CalibrationMethod::ConformalPrediction],
            conformal_prediction: Some(ConformalPredictionConfig {
                alpha,
                ..Default::default()
            }),
            ..Default::default()
        }
    }
    
    /// Create ensemble calibration configuration
    pub fn ensemble_calibration(num_models: u32) -> Self {
        Self {
            methods: vec![CalibrationMethod::EnsembleCalibration],
            ensemble_calibration: Some(EnsembleCalibrationConfig {
                num_models,
                method: EnsembleMethod::DeepEnsemble,
                bootstrap: true,
                aggregation: AggregationMethod::Average,
                diversity: None,
            }),
            ..Default::default()
        }
    }
}

impl TemperatureScalingConfig {
    fn validate(&self) -> Result<()> {
        if self.max_iter == 0 {
            return Err(NeuralForgeError::calibration(
                "Max iterations must be > 0"
            ));
        }
        
        if self.tol <= 0.0 {
            return Err(NeuralForgeError::calibration(
                "Tolerance must be > 0"
            ));
        }
        
        if self.temperature_bounds.0 >= self.temperature_bounds.1 {
            return Err(NeuralForgeError::calibration(
                "Temperature bounds must be (low, high) with low < high"
            ));
        }
        
        Ok(())
    }
}

impl ConformalPredictionConfig {
    fn validate(&self) -> Result<()> {
        if self.alpha <= 0.0 || self.alpha >= 1.0 {
            return Err(NeuralForgeError::calibration(
                "Alpha must be in (0, 1)"
            ));
        }
        
        if let Some(cal_size) = self.calibration_size {
            if cal_size <= 0.0 || cal_size >= 1.0 {
                return Err(NeuralForgeError::calibration(
                    "Calibration size must be in (0, 1)"
                ));
            }
        }
        
        Ok(())
    }
}