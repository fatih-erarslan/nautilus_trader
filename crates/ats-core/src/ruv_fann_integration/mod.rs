// ruv-FANN Neural Network Integration Module
// Production-grade implementation with 27+ neural architectures

pub mod neural_architectures;
pub mod training_pipeline;
#[cfg(feature = "ffi")]
pub mod ffi_bridge;
pub mod forecasting_engine;
pub mod gpu_acceleration;
pub mod model_persistence;
pub mod performance_benchmarks;
pub mod ats_core_integration;

// Re-export main types
pub use neural_architectures::*;
pub use training_pipeline::*;
// FFI bridge commented out for now
// pub use ffi_bridge::*;
pub use forecasting_engine::*;
pub use gpu_acceleration::*;
pub use model_persistence::*;
pub use performance_benchmarks::*;
pub use ats_core_integration::*;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Main ruv-FANN integration manager
#[derive(Clone)]
pub struct RuvFannIntegration {
    architectures: Arc<RwLock<HashMap<String, Box<dyn NeuralArchitecture>>>>,
    training_pipeline: Arc<TrainingPipeline>,
    forecasting_engine: Arc<ForecastingEngine>,
    gpu_accelerator: Arc<GpuAccelerator>,
    model_store: Arc<ModelPersistence>,
    ats_integration: Arc<AtsCoreIntegration>,
}

impl RuvFannIntegration {
    /// Create new ruv-FANN integration with all 27+ architectures
    pub async fn new() -> Result<Self, IntegrationError> {
        let mut architectures = HashMap::new();
        
        // Register all 27+ neural architectures
        Self::register_architectures(&mut architectures).await?;
        
        Ok(Self {
            architectures: Arc::new(RwLock::new(architectures)),
            training_pipeline: Arc::new(TrainingPipeline::new()),
            forecasting_engine: Arc::new(ForecastingEngine::new()),
            gpu_accelerator: Arc::new(GpuAccelerator::new().await?),
            model_store: Arc::new(ModelPersistence::new()),
            ats_integration: Arc::new(AtsCoreIntegration::new()),
        })
    }
    
    /// Register all 27+ neural architectures
    async fn register_architectures(
        architectures: &mut HashMap<String, Box<dyn NeuralArchitecture>>
    ) -> Result<(), IntegrationError> {
        // Feedforward Networks
        architectures.insert("mlp".to_string(), Box::new(MultiLayerPerceptron::new()));
        architectures.insert("deep_mlp".to_string(), Box::new(DeepMLP::new()));
        
        // Recurrent Networks
        architectures.insert("rnn".to_string(), Box::new(RecurrentNN::new()));
        architectures.insert("lstm".to_string(), Box::new(LSTM::new()));
        architectures.insert("gru".to_string(), Box::new(GRU::new()));
        architectures.insert("bi_lstm".to_string(), Box::new(BidirectionalLSTM::new()));
        
        // Convolutional Networks
        architectures.insert("cnn".to_string(), Box::new(ConvolutionalNN::new()));
        architectures.insert("resnet".to_string(), Box::new(ResNet::new()));
        architectures.insert("densenet".to_string(), Box::new(DenseNet::new()));
        architectures.insert("mobilenet".to_string(), Box::new(MobileNet::new()));
        
        // Transformer Networks
        architectures.insert("transformer".to_string(), Box::new(Transformer::new()));
        architectures.insert("bert".to_string(), Box::new(BERT::new()));
        architectures.insert("gpt".to_string(), Box::new(GPT::new()));
        
        // Time Series Specialized
        architectures.insert("nhits".to_string(), Box::new(NHiTS::new()));
        architectures.insert("nbeats".to_string(), Box::new(NBeats::new()));
        architectures.insert("temporal_cnn".to_string(), Box::new(TemporalCNN::new()));
        architectures.insert("wavenet".to_string(), Box::new(WaveNet::new()));
        
        // Attention Mechanisms
        architectures.insert("attention".to_string(), Box::new(AttentionNet::new()));
        architectures.insert("self_attention".to_string(), Box::new(SelfAttention::new()));
        architectures.insert("multi_head_attention".to_string(), Box::new(MultiHeadAttention::new()));
        
        // Advanced Architectures
        architectures.insert("neural_ode".to_string(), Box::new(NeuralODE::new()));
        architectures.insert("graph_neural_net".to_string(), Box::new(GraphNeuralNet::new()));
        architectures.insert("variational_autoencoder".to_string(), Box::new(VAE::new()));
        architectures.insert("generative_adversarial".to_string(), Box::new(GAN::new()));
        
        // Ensemble Methods
        architectures.insert("ensemble".to_string(), Box::new(EnsembleNet::new()));
        architectures.insert("bagging".to_string(), Box::new(BaggingNet::new()));
        architectures.insert("boosting".to_string(), Box::new(BoostingNet::new()));
        
        // Specialized Financial Models
        architectures.insert("financial_lstm".to_string(), Box::new(FinancialLSTM::new()));
        
        Ok(())
    }
    
    /// Get available architectures
    pub async fn get_architectures(&self) -> Vec<String> {
        let architectures = self.architectures.read().await;
        architectures.keys().cloned().collect()
    }
    
    /// Create model with specified architecture
    pub async fn create_model(&self, 
        name: String,
        architecture: String,
        config: ModelConfig
    ) -> Result<ModelId, IntegrationError> {
        let architectures = self.architectures.read().await;
        let arch = architectures.get(&architecture)
            .ok_or(IntegrationError::ArchitectureNotFound(architecture))?;
        
        let model = arch.create_model(config).await?;
        let model_id = self.model_store.save_model(name, model).await?;
        
        Ok(model_id)
    }
    
    /// Train model with real backpropagation
    pub async fn train_model(&self,
        model_id: ModelId,
        training_data: TrainingData,
        config: TrainingConfig
    ) -> Result<TrainingResult, IntegrationError> {
        let model = self.model_store.load_model(&model_id).await?;
        
        let result = self.training_pipeline.train(
            model,
            training_data,
            config,
            &self.gpu_accelerator
        ).await?;
        
        // Save trained model (clone the model Arc to keep it in result)
        self.model_store.update_model(model_id, result.model.clone()).await?;

        Ok(result)
    }
    
    /// Generate forecasts with uncertainty quantification
    pub async fn forecast(&self,
        model_id: ModelId,
        input_data: InputData,
        forecast_config: ForecastConfig
    ) -> Result<ForecastResult, IntegrationError> {
        let model = self.model_store.load_model(&model_id).await?;
        
        self.forecasting_engine.forecast(model, input_data, forecast_config).await
    }
    
    /// Integrate with ATS-Core for calibrated predictions
    pub async fn calibrated_prediction(&self,
        model_id: ModelId,
        input_data: InputData,
        calibration_config: CalibrationConfig
    ) -> Result<CalibratedPrediction, IntegrationError> {
        let model = self.model_store.load_model(&model_id).await?;
        
        self.ats_integration.calibrated_prediction(
            model,
            input_data,
            calibration_config
        ).await
    }
    
    /// Run performance benchmarks
    pub async fn benchmark(&self) -> Result<performance_benchmarks::BenchmarkResults, IntegrationError> {
        let benchmarks = PerformanceBenchmarks::new();
        benchmarks.run_all_benchmarks(&self.architectures, &self.gpu_accelerator).await
    }
}

/// Error types for ruv-FANN integration
#[derive(Debug, thiserror::Error)]
pub enum IntegrationError {
    #[error("Architecture not found: {0}")]
    ArchitectureNotFound(String),
    
    #[error("Training failed: {0}")]
    TrainingFailed(String),
    
    #[error("Forecasting failed: {0}")]
    ForecastingFailed(String),
    
    #[error("GPU acceleration failed: {0}")]
    GpuAccelerationFailed(String),
    
    #[error("Model persistence failed: {0}")]
    ModelPersistenceFailed(String),
    
    #[error("ATS-Core integration failed: {0}")]
    AtsCoreIntegrationFailed(String),
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub input_size: usize,
    pub hidden_sizes: Vec<usize>,
    pub output_size: usize,
    pub activation: ActivationFunction,
    pub dropout_rate: Option<f32>,
    pub regularization: Option<RegularizationType>,
    pub architecture_specific: serde_json::Value,
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub optimizer: OptimizerType,
    pub loss_function: LossFunction,
    pub device: DeviceType,
    pub early_stopping: Option<EarlyStoppingConfig>,
    pub scheduler: Option<SchedulerConfig>,
}

/// Forecast configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastConfig {
    pub horizon: usize,
    pub confidence_intervals: Vec<f32>, // e.g., [0.8, 0.95]
    pub uncertainty_quantification: bool,
    pub monte_carlo_samples: Option<usize>,
}

/// Model identifier
pub type ModelId = String;

/// Training data structure
#[derive(Debug, Clone)]
pub struct TrainingData {
    pub features: Vec<Vec<f32>>,
    pub targets: Vec<Vec<f32>>,
    pub validation_split: Option<f32>,
}

/// Input data for prediction/forecasting
#[derive(Debug, Clone)]
pub struct InputData {
    pub features: Vec<Vec<f32>>,
    pub sequence_length: Option<usize>, // for time series
}

/// Training result
#[derive(Clone)]
pub struct TrainingResult {
    pub model: Arc<dyn NeuralModel>,
    pub loss_history: Vec<f32>,
    pub accuracy_history: Vec<f32>,
    pub training_time: std::time::Duration,
    pub final_loss: f32,
    pub final_accuracy: f32,
}

impl std::fmt::Debug for TrainingResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainingResult")
            .field("model", &"<dyn NeuralModel>")
            .field("loss_history", &self.loss_history)
            .field("accuracy_history", &self.accuracy_history)
            .field("training_time", &self.training_time)
            .field("final_loss", &self.final_loss)
            .field("final_accuracy", &self.final_accuracy)
            .finish()
    }
}

/// Forecast result with uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResult {
    pub predictions: Vec<f32>,
    pub confidence_intervals: HashMap<String, (Vec<f32>, Vec<f32>)>, // (lower, upper)
    pub uncertainty: Vec<f32>,
    pub forecast_horizon: usize,
}

/// Calibrated prediction from ATS-Core integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibratedPrediction {
    pub prediction: f32,
    pub calibrated_prediction: f32,
    pub confidence: f32,
    pub temperature: f32,
    pub ats_score: f32,
}

/// Benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub architecture_benchmarks: HashMap<String, ArchitectureBenchmark>,
    pub gpu_benchmarks: GpuBenchmarkResult,
    pub training_benchmarks: TrainingBenchmarkResult,
    pub inference_benchmarks: InferenceBenchmarkResult,
}

/// Activation functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU,
    ELU,
    GELU,
    Swish,
}

/// Regularization types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegularizationType {
    L1(f32),
    L2(f32),
    Dropout(f32),
    BatchNorm,
    LayerNorm,
}

/// Optimizer types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerType {
    SGD { momentum: Option<f32> },
    Adam { beta1: f32, beta2: f32, eps: f32 },
    AdamW { weight_decay: f32 },
    RMSprop { alpha: f32 },
}

/// Loss functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LossFunction {
    MSE,
    MAE,
    Huber { delta: f32 },
    CrossEntropy,
    BinaryCrossEntropy,
    KLDivergence,
}

/// Device types for computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceType {
    CPU,
    GPU,
    WebGL,
    WASM,
    Auto,
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    pub patience: usize,
    pub min_delta: f32,
    pub monitor: String, // "loss" or "accuracy"
}

/// Learning rate scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub scheduler_type: SchedulerType,
    pub step_size: Option<usize>,
    pub gamma: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulerType {
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
}

/// Calibration configuration for ATS-Core integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    pub temperature_scaling: bool,
    pub platt_scaling: bool,
    pub isotonic_regression: bool,
    pub conformal_prediction: bool,
}