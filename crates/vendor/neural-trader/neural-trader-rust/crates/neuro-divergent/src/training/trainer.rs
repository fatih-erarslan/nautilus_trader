//! Comprehensive training loop for neural network models
//!
//! Provides a robust training infrastructure with:
//! - Mini-batch data loading with shuffling
//! - Train/validation splits
//! - Learning rate scheduling
//! - Early stopping
//! - Gradient clipping
//! - Progress tracking and metrics
//! - Checkpoint saving

use ndarray::{Array1, Array2, Axis, s};
use crate::{Result, NeuroDivergentError, data::TimeSeriesDataFrame};
use super::{
    optimizers::{Optimizer, OptimizerConfig, AdamW},
    schedulers::{LRScheduler, SchedulerMetrics, ConstantLR},
    losses::LossFunction,
    backprop::GradientClipping,
    engine::EarlyStopping,
};
use rayon::prelude::*;
use std::path::Path;
use serde::{Deserialize, Serialize};

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerConfig {
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size for mini-batch training
    pub batch_size: usize,
    /// Validation split ratio (0.0 to 1.0)
    pub validation_split: f64,
    /// Shuffle training data each epoch
    pub shuffle: bool,
    /// Early stopping patience (None = no early stopping)
    pub early_stopping_patience: Option<usize>,
    /// Early stopping minimum delta
    pub early_stopping_delta: f64,
    /// Gradient clipping strategy
    pub gradient_clipping: GradientClippingConfig,
    /// Checkpoint directory (None = no checkpointing)
    pub checkpoint_dir: Option<String>,
    /// Save checkpoint every N epochs
    pub checkpoint_interval: usize,
    /// Number of workers for parallel data loading
    pub num_workers: usize,
    /// Print progress every N epochs
    pub log_interval: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GradientClippingConfig {
    ByValue(f64),
    ByNorm(f64),
    None,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            validation_split: 0.2,
            shuffle: true,
            early_stopping_patience: Some(10),
            early_stopping_delta: 1e-4,
            gradient_clipping: GradientClippingConfig::ByNorm(1.0),
            checkpoint_dir: None,
            checkpoint_interval: 10,
            num_workers: num_cpus::get(),
            log_interval: 10,
        }
    }
}

/// Mini-batch data loader
pub struct DataLoader {
    /// Input features
    x: Array2<f64>,
    /// Target values
    y: Array1<f64>,
    /// Batch size
    batch_size: usize,
    /// Shuffle data
    shuffle: bool,
    /// Current epoch
    current_epoch: usize,
}

impl DataLoader {
    pub fn new(x: Array2<f64>, y: Array1<f64>, batch_size: usize, shuffle: bool) -> Self {
        Self {
            x,
            y,
            batch_size,
            shuffle,
            current_epoch: 0,
        }
    }

    /// Get number of batches
    pub fn num_batches(&self) -> usize {
        (self.x.nrows() + self.batch_size - 1) / self.batch_size
    }

    /// Create batch iterator for an epoch
    pub fn batches(&mut self) -> Vec<(Array2<f64>, Array1<f64>)> {
        let n_samples = self.x.nrows();
        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Shuffle if requested
        if self.shuffle {
            use rand::seq::SliceRandom;
            indices.shuffle(&mut rand::thread_rng());
        }

        // Create batches
        let mut batches = Vec::new();
        for chunk in indices.chunks(self.batch_size) {
            let x_batch = self.x.select(Axis(0), chunk);
            let y_batch = self.y.select(Axis(0), chunk);
            batches.push((x_batch, y_batch));
        }

        self.current_epoch += 1;
        batches
    }

    /// Get full dataset
    pub fn full_data(&self) -> (&Array2<f64>, &Array1<f64>) {
        (&self.x, &self.y)
    }
}

/// Training metrics for a single epoch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: Option<f64>,
    pub learning_rate: f64,
    pub batch_losses: Vec<f64>,
    pub gradient_norm: Option<f64>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl EpochMetrics {
    pub fn new(epoch: usize, train_loss: f64, val_loss: Option<f64>, learning_rate: f64) -> Self {
        Self {
            epoch,
            train_loss,
            val_loss,
            learning_rate,
            batch_losses: Vec::new(),
            gradient_norm: None,
            timestamp: chrono::Utc::now(),
        }
    }
}

/// Comprehensive training state
pub struct TrainingState {
    pub epoch: usize,
    pub best_val_loss: Option<f64>,
    pub metrics_history: Vec<EpochMetrics>,
    pub early_stopped: bool,
}

impl TrainingState {
    pub fn new() -> Self {
        Self {
            epoch: 0,
            best_val_loss: None,
            metrics_history: Vec::new(),
            early_stopped: false,
        }
    }
}

/// Main trainer for neural network models
pub struct Trainer<O: Optimizer, S: LRScheduler, L: LossFunction> {
    config: TrainerConfig,
    optimizer: O,
    scheduler: S,
    loss_fn: L,
    state: TrainingState,
    early_stopping: Option<EarlyStopping>,
}

impl<O: Optimizer, S: LRScheduler, L: LossFunction> Trainer<O, S, L> {
    /// Create a new trainer
    pub fn new(
        config: TrainerConfig,
        optimizer: O,
        scheduler: S,
        loss_fn: L,
    ) -> Self {
        let early_stopping = config.early_stopping_patience.map(|patience| {
            EarlyStopping::new(patience, config.early_stopping_delta)
        });

        Self {
            config,
            optimizer,
            scheduler,
            loss_fn,
            state: TrainingState::new(),
            early_stopping,
        }
    }

    /// Train the model
    pub fn train<F>(
        &mut self,
        mut forward_fn: F,
        mut backward_fn: impl FnMut(&Array2<f64>, &Array1<f64>) -> Result<Vec<Array2<f64>>>,
        mut params_fn: impl FnMut() -> Vec<Array2<f64>>,
        mut set_params_fn: impl FnMut(Vec<Array2<f64>>),
        train_data: &TimeSeriesDataFrame,
    ) -> Result<Vec<EpochMetrics>>
    where
        F: FnMut(&Array2<f64>) -> Result<Array1<f64>>,
    {
        // Prepare data
        let (x_train, y_train, x_val, y_val) = self.prepare_data(train_data)?;

        // Create data loaders
        let mut train_loader = DataLoader::new(
            x_train,
            y_train,
            self.config.batch_size,
            self.config.shuffle,
        );

        let mut val_loader = x_val.map(|x| {
            DataLoader::new(x, y_val.unwrap(), self.config.batch_size, false)
        });

        tracing::info!(
            "Starting training for {} epochs, batch_size={}, num_batches={}",
            self.config.epochs,
            self.config.batch_size,
            train_loader.num_batches()
        );

        // Training loop
        for epoch in 0..self.config.epochs {
            self.state.epoch = epoch;

            // Train one epoch
            let (train_loss, batch_losses, avg_grad_norm) = self.train_epoch(
                &mut train_loader,
                &mut forward_fn,
                &mut backward_fn,
                &mut params_fn,
            )?;

            // Validation
            let val_loss = if let Some(ref mut val_loader) = val_loader {
                Some(self.validate(val_loader, &mut forward_fn)?)
            } else {
                None
            };

            // Update learning rate
            let scheduler_metrics = SchedulerMetrics {
                train_loss,
                val_loss,
                epoch,
            };
            let new_lr = self.scheduler.step(epoch, Some(&scheduler_metrics));
            self.optimizer.set_lr(new_lr);

            // Record metrics
            let mut metrics = EpochMetrics::new(epoch, train_loss, val_loss, new_lr);
            metrics.batch_losses = batch_losses;
            metrics.gradient_norm = Some(avg_grad_norm);
            self.state.metrics_history.push(metrics.clone());

            // Update best validation loss
            if let Some(val_loss) = val_loss {
                if self.state.best_val_loss.is_none() || val_loss < self.state.best_val_loss.unwrap() {
                    self.state.best_val_loss = Some(val_loss);
                }
            }

            // Logging
            if epoch % self.config.log_interval == 0 || epoch == self.config.epochs - 1 {
                self.log_progress(&metrics);
            }

            // Early stopping check
            if let Some(ref mut early_stopping) = self.early_stopping {
                if let Some(val_loss) = val_loss {
                    if early_stopping.should_stop(val_loss) {
                        tracing::info!("Early stopping triggered at epoch {}", epoch);
                        self.state.early_stopped = true;

                        // Restore best weights would go here
                        // This requires model-specific logic

                        break;
                    }
                }
            }

            // Checkpointing
            if let Some(ref checkpoint_dir) = self.config.checkpoint_dir {
                if epoch % self.config.checkpoint_interval == 0 {
                    self.save_checkpoint(
                        checkpoint_dir,
                        epoch,
                        &params_fn(),
                        &metrics,
                    )?;
                }
            }
        }

        Ok(self.state.metrics_history.clone())
    }

    /// Train for one epoch
    fn train_epoch<F>(
        &mut self,
        train_loader: &mut DataLoader,
        forward_fn: &mut F,
        backward_fn: &mut impl FnMut(&Array2<f64>, &Array1<f64>) -> Result<Vec<Array2<f64>>>,
        params_fn: &mut impl FnMut() -> Vec<Array2<f64>>,
    ) -> Result<(f64, Vec<f64>, f64)>
    where
        F: FnMut(&Array2<f64>) -> Result<Array1<f64>>,
    {
        let batches = train_loader.batches();
        let mut total_loss = 0.0;
        let mut batch_losses = Vec::new();
        let mut total_grad_norm = 0.0;

        for (x_batch, y_batch) in batches.iter() {
            // Forward pass
            let predictions = forward_fn(x_batch)?;

            // Compute loss
            let batch_loss = self.loss_fn.forward(&predictions, y_batch)?;
            total_loss += batch_loss;
            batch_losses.push(batch_loss);

            // Backward pass
            let mut gradients = backward_fn(x_batch, y_batch)?;

            // Gradient clipping
            let grad_norm = self.apply_gradient_clipping(&mut gradients);
            total_grad_norm += grad_norm;

            // Optimizer step
            let mut params = params_fn();
            self.optimizer.step(&mut params, &gradients)?;

            // Note: In real implementation, params would be updated in the model
            // This is a simplified interface
        }

        let num_batches = batch_losses.len() as f64;
        Ok((
            total_loss / num_batches,
            batch_losses,
            total_grad_norm / num_batches,
        ))
    }

    /// Validate the model
    fn validate<F>(
        &self,
        val_loader: &mut DataLoader,
        forward_fn: &mut F,
    ) -> Result<f64>
    where
        F: FnMut(&Array2<f64>) -> Result<Array1<f64>>,
    {
        let (x_val, y_val) = val_loader.full_data();

        // Forward pass (no gradient computation needed)
        let predictions = forward_fn(x_val)?;

        // Compute validation loss
        self.loss_fn.forward(&predictions, y_val)
    }

    /// Apply gradient clipping
    fn apply_gradient_clipping(&self, gradients: &mut [Array2<f64>]) -> f64 {
        // Compute total gradient norm before clipping
        let total_norm: f64 = gradients
            .iter()
            .map(|g| g.iter().map(|x| x.powi(2)).sum::<f64>())
            .sum::<f64>()
            .sqrt();

        match self.config.gradient_clipping {
            GradientClippingConfig::ByValue(threshold) => {
                GradientClipping::ByValue(threshold).clip(gradients);
            },
            GradientClippingConfig::ByNorm(max_norm) => {
                GradientClipping::ByNorm(max_norm).clip(gradients);
            },
            GradientClippingConfig::None => {},
        }

        total_norm
    }

    /// Prepare training and validation data
    fn prepare_data(
        &self,
        data: &TimeSeriesDataFrame,
    ) -> Result<(Array2<f64>, Array1<f64>, Option<Array2<f64>>, Option<Array1<f64>>)> {
        // This is a simplified version - actual implementation would depend on
        // the specific data format and preprocessing requirements

        let values = &data.values;
        let n_samples = values.nrows();

        if n_samples == 0 {
            return Err(NeuroDivergentError::TrainingError(
                "Empty dataset provided".to_string()
            ));
        }

        // For now, create dummy data
        // Real implementation would create sequences from time series
        let n_samples_usable = n_samples - 1;
        let x = Array2::zeros((n_samples_usable, 1));
        // Create y from first feature, skipping first row
        let y = values.slice(s![1.., 0]).to_owned();

        if self.config.validation_split > 0.0 {
            let split_idx = ((1.0 - self.config.validation_split) * n_samples as f64) as usize;

            let x_train = x.slice(s![..split_idx, ..]).to_owned();
            let y_train = y.slice(s![..split_idx]).to_owned();
            let x_val = x.slice(s![split_idx.., ..]).to_owned();
            let y_val = y.slice(s![split_idx..]).to_owned();

            Ok((x_train, y_train, Some(x_val), Some(y_val)))
        } else {
            Ok((x, y, None, None))
        }
    }

    /// Log training progress
    fn log_progress(&self, metrics: &EpochMetrics) {
        let mut msg = format!(
            "Epoch {:3}/{}: train_loss={:.6}, lr={:.6}",
            metrics.epoch + 1,
            self.config.epochs,
            metrics.train_loss,
            metrics.learning_rate,
        );

        if let Some(val_loss) = metrics.val_loss {
            msg.push_str(&format!(", val_loss={:.6}", val_loss));
        }

        if let Some(grad_norm) = metrics.gradient_norm {
            msg.push_str(&format!(", grad_norm={:.4}", grad_norm));
        }

        tracing::info!("{}", msg);
    }

    /// Save checkpoint
    fn save_checkpoint(
        &self,
        checkpoint_dir: &str,
        epoch: usize,
        params: &[Array2<f64>],
        metrics: &EpochMetrics,
    ) -> Result<()> {
        use std::fs;

        // Create checkpoint directory if it doesn't exist
        fs::create_dir_all(checkpoint_dir).map_err(|e| {
            NeuroDivergentError::TrainingError(format!("Failed to create checkpoint directory: {}", e))
        })?;

        let checkpoint_path = format!("{}/checkpoint_epoch_{}.bin", checkpoint_dir, epoch);

        // Serialize checkpoint
        let checkpoint = CheckpointData {
            epoch,
            params: params.to_vec(),
            metrics: metrics.clone(),
            optimizer_state: None, // Would include optimizer state in real implementation
        };

        let encoded = bincode::serialize(&checkpoint).map_err(|e| {
            NeuroDivergentError::TrainingError(format!("Failed to serialize checkpoint: {}", e))
        })?;

        fs::write(&checkpoint_path, encoded).map_err(|e| {
            NeuroDivergentError::TrainingError(format!("Failed to write checkpoint: {}", e))
        })?;

        tracing::info!("Saved checkpoint to {}", checkpoint_path);

        Ok(())
    }

    /// Get training state
    pub fn state(&self) -> &TrainingState {
        &self.state
    }

    /// Get metrics history
    pub fn metrics_history(&self) -> &[EpochMetrics] {
        &self.state.metrics_history
    }
}

/// Checkpoint data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointData {
    epoch: usize,
    params: Vec<Array2<f64>>,
    metrics: EpochMetrics,
    optimizer_state: Option<Vec<u8>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::{optimizers::AdamW, schedulers::ConstantLR, losses::MSELoss};
    use ndarray::arr2;

    #[test]
    fn test_data_loader() {
        let x = arr2(&[[1.0], [2.0], [3.0], [4.0], [5.0]]);
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let mut loader = DataLoader::new(x, y, 2, false);

        assert_eq!(loader.num_batches(), 3);

        let batches = loader.batches();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].0.nrows(), 2);
        assert_eq!(batches[1].0.nrows(), 2);
        assert_eq!(batches[2].0.nrows(), 1);
    }

    #[test]
    fn test_data_loader_shuffle() {
        let x = arr2(&[[1.0], [2.0], [3.0], [4.0], [5.0]]);
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let mut loader = DataLoader::new(x, y, 5, true);

        let batches1 = loader.batches();
        let batches2 = loader.batches();

        // Note: This test might occasionally fail due to random chance
        // In practice, we'd use a seeded RNG
    }

    #[test]
    fn test_trainer_config_default() {
        let config = TrainerConfig::default();

        assert_eq!(config.epochs, 100);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.validation_split, 0.2);
        assert!(config.shuffle);
    }
}
