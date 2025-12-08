//! Training configuration

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use crate::error::{Result, NeuralForgeError};

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of training epochs
    pub epochs: u32,
    
    /// Batch size
    pub batch_size: usize,
    
    /// Validation batch size (if different from training)
    pub val_batch_size: Option<usize>,
    
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: u32,
    
    /// Maximum gradient norm for clipping
    pub max_grad_norm: Option<f64>,
    
    /// Mixed precision training
    pub mixed_precision: MixedPrecisionConfig,
    
    /// Early stopping configuration
    pub early_stopping: Option<EarlyStoppingConfig>,
    
    /// Validation configuration
    pub validation: ValidationConfig,
    
    /// Checkpointing configuration
    pub checkpoint: CheckpointConfig,
    
    /// Loss function configuration
    pub loss: LossConfig,
    
    /// Metrics to track
    pub metrics: Vec<MetricConfig>,
    
    /// Training callbacks
    pub callbacks: Vec<CallbackConfig>,
    
    /// Regularization techniques
    pub regularization: RegularizationConfig,
    
    /// Data augmentation
    pub augmentation: Option<AugmentationConfig>,
    
    /// Training mode
    pub mode: TrainingMode,
    
    /// Resume from checkpoint
    pub resume_from: Option<PathBuf>,
    
    /// Seed for reproducibility
    pub seed: Option<u64>,
}

/// Mixed precision training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionConfig {
    /// Enable mixed precision
    pub enabled: bool,
    
    /// Loss scaling strategy
    pub loss_scaling: LossScalingConfig,
    
    /// Optimization level
    pub opt_level: OptLevel,
}

/// Loss scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LossScalingConfig {
    /// Fixed loss scaling
    Fixed { scale: f64 },
    
    /// Dynamic loss scaling
    Dynamic {
        init_scale: f64,
        growth_factor: f64,
        backoff_factor: f64,
        growth_interval: u32,
    },
    
    /// Automatic loss scaling
    Auto,
}

/// Optimization levels for mixed precision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptLevel {
    O0, // FP32 training
    O1, // Conservative mixed precision
    O2, // Fast mixed precision
    O3, // FP16 training
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Metric to monitor
    pub monitor: String,
    
    /// Minimum change to qualify as improvement
    pub min_delta: f64,
    
    /// Number of epochs with no improvement to wait
    pub patience: u32,
    
    /// Whether higher values are better
    pub maximize: bool,
    
    /// Restore best weights on early stop
    pub restore_best_weights: bool,
}

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Validation frequency (epochs)
    pub frequency: u32,
    
    /// Validation split (if no separate validation set)
    pub split: Option<f64>,
    
    /// Stratified split for classification
    pub stratified: bool,
    
    /// Cross-validation configuration
    pub cross_validation: Option<CrossValidationConfig>,
    
    /// Validation metrics
    pub metrics: Vec<MetricConfig>,
}

/// Cross-validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationConfig {
    /// Number of folds
    pub n_folds: u32,
    
    /// Stratified k-fold
    pub stratified: bool,
    
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    
    /// Shuffle data before splitting
    pub shuffle: bool,
}

/// Checkpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Checkpoint directory
    pub dir: PathBuf,
    
    /// Checkpoint frequency (epochs)
    pub frequency: u32,
    
    /// Keep only best checkpoint
    pub save_best_only: bool,
    
    /// Metric to use for best checkpoint
    pub monitor: String,
    
    /// Whether higher is better for monitor metric
    pub maximize: bool,
    
    /// Maximum number of checkpoints to keep
    pub max_checkpoints: Option<u32>,
    
    /// Save optimizer state
    pub save_optimizer: bool,
    
    /// Save scheduler state
    pub save_scheduler: bool,
    
    /// Checkpoint format
    pub format: CheckpointFormat,
}

/// Checkpoint formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CheckpointFormat {
    /// Native Candle format
    Candle,
    
    /// ONNX format
    Onnx,
    
    /// TensorRT format
    TensorRT,
    
    /// Custom format
    Custom { name: String },
}

/// Loss function configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LossConfig {
    /// Mean Squared Error
    MSE,
    
    /// Mean Absolute Error
    MAE,
    
    /// Cross Entropy
    CrossEntropy {
        weight: Option<Vec<f64>>,
        label_smoothing: Option<f64>,
    },
    
    /// Binary Cross Entropy
    BCE {
        pos_weight: Option<f64>,
        label_smoothing: Option<f64>,
    },
    
    /// Focal Loss
    Focal {
        alpha: f64,
        gamma: f64,
    },
    
    /// Huber Loss
    Huber {
        delta: f64,
    },
    
    /// Contrastive Loss
    Contrastive {
        margin: f64,
    },
    
    /// Triplet Loss
    Triplet {
        margin: f64,
        p: f64,
        eps: f64,
    },
    
    /// Custom loss
    Custom {
        name: String,
        params: std::collections::HashMap<String, serde_json::Value>,
    },
}

/// Metric configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricConfig {
    /// Accuracy
    Accuracy {
        threshold: Option<f64>,
        top_k: Option<usize>,
    },
    
    /// Precision
    Precision {
        average: AverageType,
        labels: Option<Vec<usize>>,
    },
    
    /// Recall
    Recall {
        average: AverageType,
        labels: Option<Vec<usize>>,
    },
    
    /// F1 Score
    F1 {
        average: AverageType,
        labels: Option<Vec<usize>>,
    },
    
    /// Area Under Curve
    AUC {
        multi_class: Option<String>,
    },
    
    /// Mean Average Precision
    MAP {
        k: Option<usize>,
    },
    
    /// BLEU Score (for text generation)
    BLEU {
        n_gram: usize,
    },
    
    /// Custom metric
    Custom {
        name: String,
        params: std::collections::HashMap<String, serde_json::Value>,
    },
}

/// Averaging types for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AverageType {
    Micro,
    Macro,
    Weighted,
    None,
}

/// Callback configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CallbackConfig {
    /// Learning rate finder
    LRFinder {
        start_lr: f64,
        end_lr: f64,
        num_iter: u32,
    },
    
    /// Model pruning
    Pruning {
        sparsity: f64,
        structured: bool,
    },
    
    /// Gradient clipping
    GradientClipping {
        max_norm: f64,
        norm_type: f64,
    },
    
    /// Custom callback
    Custom {
        name: String,
        params: std::collections::HashMap<String, serde_json::Value>,
    },
}

/// Regularization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationConfig {
    /// Weight decay
    pub weight_decay: f64,
    
    /// Dropout configurations
    pub dropout: DropoutConfig,
    
    /// Batch normalization
    pub batch_norm: BatchNormConfig,
    
    /// Layer normalization
    pub layer_norm: LayerNormConfig,
    
    /// Label smoothing
    pub label_smoothing: Option<f64>,
    
    /// Mixup augmentation
    pub mixup: Option<MixupConfig>,
    
    /// CutMix augmentation
    pub cutmix: Option<CutMixConfig>,
}

/// Dropout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropoutConfig {
    /// Standard dropout rate
    pub rate: f64,
    
    /// Scheduled dropout
    pub scheduled: Option<ScheduledDropoutConfig>,
}

/// Scheduled dropout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledDropoutConfig {
    /// Initial dropout rate
    pub initial_rate: f64,
    
    /// Final dropout rate
    pub final_rate: f64,
    
    /// Schedule type
    pub schedule: DropoutSchedule,
}

/// Dropout schedule types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DropoutSchedule {
    Linear,
    Exponential { decay_rate: f64 },
    Cosine,
}

/// Batch normalization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchNormConfig {
    /// Enable batch normalization
    pub enabled: bool,
    
    /// Momentum for running statistics
    pub momentum: f64,
    
    /// Epsilon for numerical stability
    pub eps: f64,
    
    /// Affine transformation
    pub affine: bool,
}

/// Layer normalization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNormConfig {
    /// Enable layer normalization
    pub enabled: bool,
    
    /// Epsilon for numerical stability
    pub eps: f64,
    
    /// Elementwise affine
    pub elementwise_affine: bool,
}

/// Mixup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixupConfig {
    /// Alpha parameter for Beta distribution
    pub alpha: f64,
    
    /// Probability of applying mixup
    pub prob: f64,
}

/// CutMix configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CutMixConfig {
    /// Alpha parameter for Beta distribution
    pub alpha: f64,
    
    /// Probability of applying cutmix
    pub prob: f64,
}

/// Data augmentation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationConfig {
    /// Augmentation techniques
    pub techniques: Vec<AugmentationTechnique>,
    
    /// Probability of applying augmentation
    pub prob: f64,
    
    /// Number of augmentations to apply
    pub num_ops: Option<usize>,
    
    /// Magnitude of augmentations
    pub magnitude: Option<f64>,
}

/// Augmentation techniques
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AugmentationTechnique {
    /// Gaussian noise
    GaussianNoise { std: f64 },
    
    /// Random scaling
    RandomScale { scale_range: (f64, f64) },
    
    /// Random rotation
    RandomRotation { degrees: f64 },
    
    /// Time warping (for time series)
    TimeWarping { sigma: f64 },
    
    /// Magnitude warping (for time series)
    MagnitudeWarping { sigma: f64 },
    
    /// Custom augmentation
    Custom {
        name: String,
        params: std::collections::HashMap<String, serde_json::Value>,
    },
}

/// Training modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainingMode {
    /// Standard training
    Standard,
    
    /// Adversarial training
    Adversarial {
        epsilon: f64,
        alpha: f64,
        num_iter: u32,
    },
    
    /// Self-supervised learning
    SelfSupervised {
        pretext_task: PreTextTask,
    },
    
    /// Transfer learning
    Transfer {
        pretrained_path: PathBuf,
        freeze_layers: Option<Vec<String>>,
        fine_tune_lr: Option<f64>,
    },
    
    /// Curriculum learning
    Curriculum {
        difficulty_fn: String,
        pacing_fn: String,
    },
    
    /// Meta learning
    Meta {
        inner_lr: f64,
        inner_steps: u32,
        meta_lr: f64,
    },
}

/// Pretext tasks for self-supervised learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreTextTask {
    /// Autoencoding
    Autoencoder,
    
    /// Contrastive learning
    Contrastive { temperature: f64 },
    
    /// Masked language modeling
    MaskedLM { mask_prob: f64 },
    
    /// Next sentence prediction
    NextSentence,
    
    /// Rotation prediction
    Rotation,
    
    /// Custom pretext task
    Custom {
        name: String,
        params: std::collections::HashMap<String, serde_json::Value>,
    },
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            val_batch_size: None,
            gradient_accumulation_steps: 1,
            max_grad_norm: Some(1.0),
            mixed_precision: MixedPrecisionConfig::default(),
            early_stopping: Some(EarlyStoppingConfig::default()),
            validation: ValidationConfig::default(),
            checkpoint: CheckpointConfig::default(),
            loss: LossConfig::BCE { pos_weight: None, label_smoothing: None },
            metrics: vec![
                MetricConfig::Accuracy { threshold: None, top_k: None },
                MetricConfig::F1 { average: AverageType::Macro, labels: None },
            ],
            callbacks: vec![],
            regularization: RegularizationConfig::default(),
            augmentation: None,
            mode: TrainingMode::Standard,
            resume_from: None,
            seed: Some(42),
        }
    }
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            loss_scaling: LossScalingConfig::Dynamic {
                init_scale: 2_f64.powi(16),
                growth_factor: 2.0,
                backoff_factor: 0.5,
                growth_interval: 2000,
            },
            opt_level: OptLevel::O1,
        }
    }
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            monitor: "val_loss".to_string(),
            min_delta: 0.0,
            patience: 10,
            maximize: false,
            restore_best_weights: true,
        }
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            frequency: 1,
            split: Some(0.2),
            stratified: true,
            cross_validation: None,
            metrics: vec![
                MetricConfig::Accuracy { threshold: None, top_k: None },
                MetricConfig::F1 { average: AverageType::Macro, labels: None },
            ],
        }
    }
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            dir: PathBuf::from("./checkpoints"),
            frequency: 1,
            save_best_only: true,
            monitor: "val_loss".to_string(),
            maximize: false,
            max_checkpoints: Some(5),
            save_optimizer: true,
            save_scheduler: true,
            format: CheckpointFormat::Candle,
        }
    }
}

impl Default for RegularizationConfig {
    fn default() -> Self {
        Self {
            weight_decay: 0.01,
            dropout: DropoutConfig {
                rate: 0.1,
                scheduled: None,
            },
            batch_norm: BatchNormConfig {
                enabled: false,
                momentum: 0.1,
                eps: 1e-5,
                affine: true,
            },
            layer_norm: LayerNormConfig {
                enabled: true,
                eps: 1e-5,
                elementwise_affine: true,
            },
            label_smoothing: None,
            mixup: None,
            cutmix: None,
        }
    }
}

impl TrainingConfig {
    /// Validate training configuration
    pub fn validate(&self) -> Result<()> {
        if self.epochs == 0 {
            return Err(NeuralForgeError::config("Epochs must be > 0"));
        }
        
        if self.batch_size == 0 {
            return Err(NeuralForgeError::config("Batch size must be > 0"));
        }
        
        if self.gradient_accumulation_steps == 0 {
            return Err(NeuralForgeError::config("Gradient accumulation steps must be > 0"));
        }
        
        if let Some(max_norm) = self.max_grad_norm {
            if max_norm <= 0.0 {
                return Err(NeuralForgeError::config("Max gradient norm must be > 0"));
            }
        }
        
        if let Some(ref validation) = self.validation.split {
            if *validation <= 0.0 || *validation >= 1.0 {
                return Err(NeuralForgeError::config("Validation split must be in (0, 1)"));
            }
        }
        
        Ok(())
    }
    
    /// Builder methods
    pub fn with_epochs(mut self, epochs: u32) -> Self {
        self.epochs = epochs;
        self
    }
    
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
    
    pub fn with_learning_rate_finder(mut self) -> Self {
        self.callbacks.push(CallbackConfig::LRFinder {
            start_lr: 1e-8,
            end_lr: 1e-1,
            num_iter: 100,
        });
        self
    }
    
    pub fn with_early_stopping(mut self, patience: u32) -> Self {
        self.early_stopping = Some(EarlyStoppingConfig {
            patience,
            ..EarlyStoppingConfig::default()
        });
        self
    }
    
    pub fn with_mixed_precision(mut self, enabled: bool) -> Self {
        self.mixed_precision.enabled = enabled;
        self
    }
}