//! Training engine and utilities for neural network models
//!
//! This module provides a complete training infrastructure including:
//! - **Backpropagation**: Automatic differentiation with gradient tape
//! - **Optimizers**: AdamW, SGD (with Nesterov momentum), RMSprop
//! - **Loss Functions**: MSE, MAE, Huber, Quantile, MAPE, SMAPE, Weighted
//! - **Learning Rate Schedulers**: Cosine annealing, Warmup, Step decay, Reduce on plateau
//! - **Training Loop**: Mini-batch processing, validation, early stopping, checkpointing
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use neuro_divergent::training::{
//!     optimizers::{AdamW, OptimizerConfig},
//!     schedulers::CosineAnnealingLR,
//!     losses::MSELoss,
//!     trainer::{Trainer, TrainerConfig},
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Configure optimizer
//! let optimizer = AdamW::new(OptimizerConfig {
//!     learning_rate: 0.001,
//!     weight_decay: 0.0001,
//!     epsilon: 1e-8,
//! });
//!
//! // Configure learning rate scheduler
//! let scheduler = CosineAnnealingLR::new(0.001, 100, 0.0);
//!
//! // Configure trainer
//! let config = TrainerConfig {
//!     epochs: 100,
//!     batch_size: 32,
//!     validation_split: 0.2,
//!     ..Default::default()
//! };
//!
//! let trainer = Trainer::new(config, optimizer, scheduler, MSELoss);
//! # Ok(())
//! # }
//! ```
//!
//! ## Architecture
//!
//! ### Backpropagation Engine
//!
//! The gradient tape records operations during forward pass and computes
//! gradients during backward pass using automatic differentiation:
//!
//! ```rust,no_run
//! use neuro_divergent::training::backprop::{GradientTape, Activation};
//! use ndarray::arr2;
//!
//! let mut tape = GradientTape::new();
//! tape.record();
//!
//! // Forward pass
//! let a = arr2(&[[1.0, 2.0]]);
//! let b = arr2(&[[3.0], [4.0]]);
//! let c = tape.matmul(&a, &b, ("a".into(), "b".into()), "c".into());
//!
//! // Backward pass
//! let grad_c = arr2(&[[1.0]]);
//! tape.backward(grad_c, "c".into()).unwrap();
//!
//! // Get gradients
//! let grad_a = tape.get_gradient("a");
//! let grad_b = tape.get_gradient("b");
//! ```
//!
//! ### Optimizers
//!
//! All optimizers implement the `Optimizer` trait and support learning rate scheduling:
//!
//! - **AdamW**: Adam with decoupled weight decay (recommended for most cases)
//! - **SGD**: Stochastic Gradient Descent with optional momentum and Nesterov acceleration
//! - **RMSprop**: Root Mean Square Propagation for non-stationary objectives
//!
//! ### Loss Functions
//!
//! Loss functions implement `LossFunction` trait with forward and backward methods:
//!
//! - **Regression**: MSE, MAE, Huber
//! - **Probabilistic**: Quantile loss for prediction intervals
//! - **Percentage**: MAPE, SMAPE
//! - **Weighted**: Apply custom weights to samples
//!
//! ### Learning Rate Schedulers
//!
//! Schedulers adjust learning rate during training:
//!
//! - **Cosine Annealing**: Smooth decay following cosine curve
//! - **Warmup + Linear/Cosine**: Gradual warmup followed by decay
//! - **Step Decay**: Reduce LR by factor every N epochs
//! - **Reduce on Plateau**: Adaptive reduction when metrics plateau
//!
//! ## Performance Optimizations
//!
//! - **SIMD**: Vectorized operations for activations and loss functions
//! - **Rayon**: Parallel batch processing across CPU cores
//! - **Mixed Precision**: Support for f32/f64 with gradient accumulation
//! - **Gradient Checkpointing**: Memory-efficient backpropagation
//!
//! ## Example: Complete Training Pipeline
//!
//! ```rust,no_run
//! use neuro_divergent::{
//!     training::{
//!         optimizers::{AdamW, OptimizerConfig},
//!         schedulers::WarmupCosineLR,
//!         losses::MSELoss,
//!         trainer::{Trainer, TrainerConfig},
//!     },
//!     data::TimeSeriesDataFrame,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Prepare data
//! let data = TimeSeriesDataFrame::from_values(vec![1.0, 2.0, 3.0, 4.0, 5.0], None)?;
//!
//! // Configure training
//! let optimizer = AdamW::new(OptimizerConfig::default());
//! let scheduler = WarmupCosineLR::new(0.001, 10, 100, 1e-6)?;
//! let config = TrainerConfig::default();
//!
//! let mut trainer = Trainer::new(config, optimizer, scheduler, MSELoss);
//!
//! // Define model functions (simplified)
//! let forward = |x| Ok(x.sum_axis(ndarray::Axis(1)));
//! let backward = |x, y| Ok(vec![]);
//! let get_params = || vec![];
//! let set_params = |p| {};
//!
//! // Train
//! // let metrics = trainer.train(forward, backward, get_params, set_params, &data)?;
//! # Ok(())
//! # }
//! ```

pub mod backprop;
pub mod optimizers;
pub mod schedulers;
pub mod losses;
pub mod trainer;
pub mod metrics;
pub mod engine;

// Legacy exports for backward compatibility
pub mod optimizer {
    pub use super::optimizers::*;
}

// Re-export main types
pub use backprop::{GradientTape, Activation, GradientClipping};
pub use optimizers::{Optimizer, OptimizerConfig, AdamW, SGD, RMSprop};
pub use schedulers::{
    LRScheduler, SchedulerMetrics,
    CosineAnnealingLR, WarmupLinearLR, WarmupCosineLR,
    StepLR, ExponentialLR, ReduceLROnPlateau, PlateauMode, ConstantLR,
};
pub use losses::{
    LossFunction, MSELoss, MAELoss, HuberLoss,
    QuantileLoss, MAPELoss, SMAPELoss, WeightedLoss,
};
pub use trainer::{
    Trainer, TrainerConfig, DataLoader,
    EpochMetrics, TrainingState,
    GradientClippingConfig,
};
pub use metrics::{mae, mse, rmse, mape, r2_score};
pub use engine::{TrainingEngine, EarlyStopping};

// Re-export for backward compatibility
// pub use crate::training::metrics::TrainingMetrics;  // Defined below instead

use serde::{Deserialize, Serialize};

/// Training metrics collected during model training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: Option<f64>,
    pub learning_rate: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl TrainingMetrics {
    pub fn new(epoch: usize, train_loss: f64, val_loss: Option<f64>, learning_rate: f64) -> Self {
        Self {
            epoch,
            train_loss,
            val_loss,
            learning_rate,
            timestamp: chrono::Utc::now(),
        }
    }
}
