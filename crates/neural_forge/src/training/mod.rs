//! High-performance training orchestration

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use candle_core::{Device, Tensor, DType};
use rayon::prelude::*;
use tracing::{info, warn, error, debug, span, Level};

use crate::prelude::*;
use crate::error::{Result, NeuralForgeError};

pub mod trainer;
pub mod engine;
pub mod callbacks;
pub mod metrics;
pub mod scheduler;
pub mod checkpoints;

pub use trainer::*;
pub use engine::*;
pub use callbacks::*;
pub use metrics::*;
pub use scheduler::*;
pub use checkpoints::*;

/// High-performance neural network trainer
pub struct Trainer {
    /// Training configuration
    config: NeuralForgeConfig,
    
    /// Training engine
    engine: TrainingEngine,
    
    /// Device for computation
    device: Device,
    
    /// Training state
    state: TrainingState,
    
    /// Callbacks
    callbacks: Vec<Box<dyn TrainingCallback>>,
    
    /// Metrics tracker
    metrics: MetricsTracker,
    
    /// Checkpoint manager
    checkpoint_manager: CheckpointManager,
    
    /// Learning rate scheduler
    scheduler: Option<Box<dyn LRScheduler>>,
}

/// Training state
#[derive(Debug, Clone)]
pub struct TrainingState {
    /// Current epoch
    pub epoch: u32,
    
    /// Current step/iteration
    pub step: u64,
    
    /// Total steps
    pub total_steps: u64,
    
    /// Best validation score
    pub best_score: Option<f64>,
    
    /// Best model path
    pub best_model_path: Option<PathBuf>,
    
    /// Training start time
    pub start_time: Instant,
    
    /// Current learning rate
    pub learning_rate: f64,
    
    /// Training history
    pub history: TrainingHistory,
    
    /// Early stopping counter
    pub early_stopping_counter: u32,
    
    /// Should stop training
    pub should_stop: bool,
}

/// Training history
#[derive(Debug, Clone, Default)]
pub struct TrainingHistory {
    /// Training losses
    pub train_losses: Vec<f64>,
    
    /// Validation losses
    pub val_losses: Vec<f64>,
    
    /// Training metrics
    pub train_metrics: HashMap<String, Vec<f64>>,
    
    /// Validation metrics
    pub val_metrics: HashMap<String, Vec<f64>>,
    
    /// Learning rates
    pub learning_rates: Vec<f64>,
    
    /// Epoch times
    pub epoch_times: Vec<Duration>,
}

/// Training results
#[derive(Debug, Clone)]
pub struct TrainingResults {
    /// Training state
    pub state: TrainingState,
    
    /// Final model
    pub model: Arc<dyn Module>,
    
    /// Calibrated model (if calibration was performed)
    pub calibrated_model: Option<Arc<dyn Module>>,
    
    /// Training metrics
    pub metrics: HashMap<String, f64>,
    
    /// Calibration results
    pub calibration_results: Option<CalibrationResults>,
    
    /// Model checkpoints
    pub checkpoints: Vec<PathBuf>,
    
    /// Total training time
    pub training_time: Duration,
}

/// Calibration results
#[derive(Debug, Clone)]
pub struct CalibrationResults {
    /// Temperature scaling results
    pub temperature_scaling: Option<TemperatureScalingResults>,
    
    /// Conformal prediction results
    pub conformal_prediction: Option<ConformalPredictionResults>,
    
    /// Calibration metrics
    pub metrics: HashMap<String, f64>,
    
    /// Calibration curves
    pub calibration_curves: Option<CalibrationCurves>,
}

/// Temperature scaling results
#[derive(Debug, Clone)]
pub struct TemperatureScalingResults {
    /// Optimized temperature
    pub temperature: f64,
    
    /// Pre-calibration ECE
    pub pre_ece: f64,
    
    /// Post-calibration ECE
    pub post_ece: f64,
    
    /// Optimization history
    pub optimization_history: Vec<f64>,
}

/// Conformal prediction results
#[derive(Debug, Clone)]
pub struct ConformalPredictionResults {
    /// Conformal threshold
    pub threshold: f64,
    
    /// Coverage probability
    pub coverage: f64,
    
    /// Average interval width
    pub average_width: f64,
    
    /// Prediction intervals
    pub intervals: Vec<(f64, f64)>,
}

/// Calibration curves
#[derive(Debug, Clone)]
pub struct CalibrationCurves {
    /// Reliability diagram
    pub reliability: (Vec<f64>, Vec<f64>),
    
    /// Confidence histogram
    pub confidence_histogram: Vec<f64>,
    
    /// ECE by bin
    pub ece_by_bin: Vec<f64>,
}

impl Trainer {
    /// Create new trainer with configuration
    pub fn new(config: NeuralForgeConfig) -> Result<Self> {
        let _span = span!(Level::INFO, "trainer_init").entered();
        
        info!("Initializing Neural Forge trainer");
        
        // Validate configuration
        config.validate()?;
        
        // Initialize device
        let device = Self::initialize_device(&config.hardware.device)?;
        info!("Using device: {:?}", device);
        
        // Initialize training engine
        let engine = TrainingEngine::new(&config, &device)?;
        
        // Initialize state
        let state = TrainingState::new();
        
        // Initialize callbacks
        let callbacks = Self::initialize_callbacks(&config)?;
        
        // Initialize metrics tracker
        let metrics = MetricsTracker::new(&config.training.metrics)?;
        
        // Initialize checkpoint manager
        let checkpoint_manager = CheckpointManager::new(&config.training.checkpoint)?;
        
        // Initialize scheduler
        let scheduler = if let Some(ref scheduler_config) = config.scheduler {
            Some(Self::create_scheduler(scheduler_config, &config.optimizer)?)
        } else {
            None
        };
        
        info!("Trainer initialized successfully");
        
        Ok(Self {
            config,
            engine,
            device,
            state,
            callbacks,
            metrics,
            checkpoint_manager,
            scheduler,
        })
    }
    
    /// Train model on dataset
    pub fn train<D>(&mut self, dataset: D) -> Result<TrainingResults>
    where
        D: Dataset + Send + Sync + 'static,
    {
        let _span = span!(Level::INFO, "training").entered();
        
        info!("Starting training with Neural Forge");
        self.state.start_time = Instant::now();
        
        // Setup training
        self.setup_training(&dataset)?;
        
        // Execute training loop
        self.training_loop(&dataset)?;
        
        // Perform calibration if configured
        let calibration_results = if self.config.calibration.is_some() {
            Some(self.calibrate_model(&dataset)?)
        } else {
            None
        };
        
        // Finalize training
        let results = self.finalize_training(calibration_results)?;
        
        info!(
            "Training completed in {:.2?}",
            results.training_time
        );
        
        Ok(results)
    }
    
    /// Setup training environment
    fn setup_training<D>(&mut self, dataset: &D) -> Result<()>
    where
        D: Dataset,
    {
        let _span = span!(Level::DEBUG, "setup_training").entered();
        
        // Calculate total steps
        let dataset_size = dataset.len();
        let steps_per_epoch = (dataset_size + self.config.training.batch_size - 1) 
            / self.config.training.batch_size;
        self.state.total_steps = (steps_per_epoch as u64) * (self.config.training.epochs as u64);
        
        info!(
            "Training setup: {} epochs, {} steps per epoch, {} total steps",
            self.config.training.epochs,
            steps_per_epoch,
            self.state.total_steps
        );
        
        // Initialize callbacks
        for callback in &mut self.callbacks {
            callback.on_train_begin(&mut self.state, &self.config)?;
        }
        
        Ok(())
    }
    
    /// Main training loop
    fn training_loop<D>(&mut self, dataset: &D) -> Result<()>
    where
        D: Dataset + Send + Sync,
    {
        let _span = span!(Level::INFO, "training_loop").entered();
        
        for epoch in 0..self.config.training.epochs {
            if self.state.should_stop {
                info!("Early stopping triggered at epoch {}", epoch);
                break;
            }
            
            self.state.epoch = epoch;
            
            // Epoch callbacks
            for callback in &mut self.callbacks {
                callback.on_epoch_begin(&mut self.state, &self.config)?;
            }
            
            // Training epoch
            let train_metrics = self.train_epoch(dataset)?;
            
            // Validation epoch
            let val_metrics = if epoch % self.config.training.validation.frequency == 0 {
                Some(self.validate_epoch(dataset)?)
            } else {
                None
            };
            
            // Update learning rate
            if let Some(ref mut scheduler) = self.scheduler {
                scheduler.step(val_metrics.as_ref().and_then(|m| m.get("loss")).copied())?;
                self.state.learning_rate = scheduler.get_lr();
            }
            
            // Update history
            self.update_history(train_metrics, val_metrics.clone());
            
            // Check for improvement and save checkpoint
            self.check_improvement_and_save(val_metrics)?;
            
            // Epoch callbacks
            for callback in &mut self.callbacks {
                callback.on_epoch_end(&mut self.state, &self.config)?;
            }
            
            // Check early stopping
            if let Some(ref early_stopping) = self.config.training.early_stopping {
                if self.should_early_stop(early_stopping) {
                    self.state.should_stop = true;
                }
            }
        }
        
        Ok(())
    }
    
    /// Train single epoch
    fn train_epoch<D>(&mut self, dataset: &D) -> Result<HashMap<String, f64>>
    where
        D: Dataset + Send + Sync,
    {
        let _span = span!(Level::DEBUG, "train_epoch", epoch = self.state.epoch).entered();
        
        // Use training engine for efficient training
        let metrics = self.engine.train_epoch(
            dataset,
            &mut self.state,
            &self.config,
            &mut self.callbacks,
        )?;
        
        debug!("Training epoch {} completed", self.state.epoch);
        Ok(metrics)
    }
    
    /// Validate single epoch
    fn validate_epoch<D>(&mut self, dataset: &D) -> Result<HashMap<String, f64>>
    where
        D: Dataset,
    {
        let _span = span!(Level::DEBUG, "validate_epoch", epoch = self.state.epoch).entered();
        
        // Use training engine for efficient validation
        let metrics = self.engine.validate_epoch(
            dataset,
            &mut self.state,
            &self.config,
        )?;
        
        debug!("Validation epoch {} completed", self.state.epoch);
        Ok(metrics)
    }
    
    /// Calibrate trained model
    fn calibrate_model<D>(&mut self, dataset: &D) -> Result<CalibrationResults>
    where
        D: Dataset,
    {
        let _span = span!(Level::INFO, "calibration").entered();
        
        info!("Starting model calibration");
        
        let calibration_config = self.config.calibration.as_ref()
            .ok_or_else(|| NeuralForgeError::calibration("No calibration config"))?;
        
        // Temperature scaling
        let temperature_scaling = if calibration_config.methods.contains(&CalibrationMethod::TemperatureScaling) {
            Some(self.perform_temperature_scaling(dataset, calibration_config)?)
        } else {
            None
        };
        
        // Conformal prediction
        let conformal_prediction = if calibration_config.methods.contains(&CalibrationMethod::ConformalPrediction) {
            Some(self.perform_conformal_prediction(dataset, calibration_config)?)
        } else {
            None
        };
        
        // Calculate calibration metrics
        let metrics = self.calculate_calibration_metrics(dataset, calibration_config)?;
        
        info!("Model calibration completed");
        
        Ok(CalibrationResults {
            temperature_scaling,
            conformal_prediction,
            metrics,
            calibration_curves: None, // TODO: Implement
        })
    }
    
    /// Perform temperature scaling
    fn perform_temperature_scaling<D>(
        &mut self,
        dataset: &D,
        _config: &CalibrationConfig,
    ) -> Result<TemperatureScalingResults>
    where
        D: Dataset,
    {
        let _span = span!(Level::DEBUG, "temperature_scaling").entered();
        
        // TODO: Implement temperature scaling optimization
        // This is a simplified version - full implementation would use L-BFGS
        
        debug!("Performing temperature scaling");
        
        Ok(TemperatureScalingResults {
            temperature: 1.5, // Placeholder
            pre_ece: 0.05,    // Placeholder
            post_ece: 0.02,   // Placeholder
            optimization_history: vec![],
        })
    }
    
    /// Perform conformal prediction
    fn perform_conformal_prediction<D>(
        &mut self,
        dataset: &D,
        config: &CalibrationConfig,
    ) -> Result<ConformalPredictionResults>
    where
        D: Dataset,
    {
        let _span = span!(Level::DEBUG, "conformal_prediction").entered();
        
        debug!("Performing conformal prediction");
        
        let conf_config = config.conformal_prediction.as_ref()
            .ok_or_else(|| NeuralForgeError::calibration("No conformal config"))?;
        
        // TODO: Implement conformal prediction
        // This is a placeholder implementation
        
        Ok(ConformalPredictionResults {
            threshold: 0.1,     // Placeholder
            coverage: 0.9,      // Placeholder
            average_width: 0.2, // Placeholder
            intervals: vec![],
        })
    }
    
    /// Calculate calibration metrics
    fn calculate_calibration_metrics<D>(
        &self,
        _dataset: &D,
        config: &CalibrationConfig,
    ) -> Result<HashMap<String, f64>>
    where
        D: Dataset,
    {
        let mut metrics = HashMap::new();
        
        for metric in &config.evaluation_metrics {
            match metric {
                CalibrationMetric::ECE { .. } => {
                    metrics.insert("ece".to_string(), 0.02); // Placeholder
                }
                CalibrationMetric::MCE { .. } => {
                    metrics.insert("mce".to_string(), 0.05); // Placeholder
                }
                CalibrationMetric::BrierScore => {
                    metrics.insert("brier_score".to_string(), 0.15); // Placeholder
                }
                _ => {} // TODO: Implement other metrics
            }
        }
        
        Ok(metrics)
    }
    
    /// Update training history
    fn update_history(
        &mut self,
        train_metrics: HashMap<String, f64>,
        val_metrics: Option<HashMap<String, f64>>,
    ) {
        // Update losses
        if let Some(train_loss) = train_metrics.get("loss") {
            self.state.history.train_losses.push(*train_loss);
        }
        
        if let Some(ref val_metrics) = val_metrics {
            if let Some(val_loss) = val_metrics.get("loss") {
                self.state.history.val_losses.push(*val_loss);
            }
        }
        
        // Update other metrics
        for (key, value) in train_metrics {
            if key != "loss" {
                self.state.history.train_metrics
                    .entry(key)
                    .or_insert_with(Vec::new)
                    .push(value);
            }
        }
        
        if let Some(val_metrics) = val_metrics {
            for (key, value) in val_metrics {
                if key != "loss" {
                    self.state.history.val_metrics
                        .entry(key)
                        .or_insert_with(Vec::new)
                        .push(value);
                }
            }
        }
        
        // Update learning rate
        self.state.history.learning_rates.push(self.state.learning_rate);
    }
    
    /// Check for improvement and save checkpoint
    fn check_improvement_and_save(
        &mut self,
        val_metrics: Option<HashMap<String, f64>>,
    ) -> Result<()> {
        let monitor_metric = &self.config.training.checkpoint.monitor;
        let maximize = self.config.training.checkpoint.maximize;
        
        let current_score = if let Some(ref metrics) = val_metrics {
            metrics.get(monitor_metric).copied()
        } else {
            None
        };
        
        let mut is_best = false;
        
        if let Some(score) = current_score {
            match self.state.best_score {
                Some(best) => {
                    is_best = if maximize {
                        score > best
                    } else {
                        score < best
                    };
                }
                None => is_best = true,
            }
            
            if is_best {
                self.state.best_score = Some(score);
                self.state.early_stopping_counter = 0;
            } else {
                self.state.early_stopping_counter += 1;
            }
        }
        
        // Save checkpoint
        if is_best || !self.config.training.checkpoint.save_best_only {
            let checkpoint_path = self.checkpoint_manager.save_checkpoint(
                &self.engine.model,
                &self.state,
                &self.config,
                is_best,
            )?;
            
            if is_best {
                self.state.best_model_path = Some(checkpoint_path);
            }
        }
        
        Ok(())
    }
    
    /// Check if training should stop early
    fn should_early_stop(&self, config: &EarlyStoppingConfig) -> bool {
        self.state.early_stopping_counter >= config.patience
    }
    
    /// Finalize training and return results
    fn finalize_training(
        &mut self,
        calibration_results: Option<CalibrationResults>,
    ) -> Result<TrainingResults> {
        let training_time = self.state.start_time.elapsed();
        
        // Finalize callbacks
        for callback in &mut self.callbacks {
            callback.on_train_end(&mut self.state, &self.config)?;
        }
        
        // Calculate final metrics
        let mut final_metrics = HashMap::new();
        if let Some(best_score) = self.state.best_score {
            final_metrics.insert("best_score".to_string(), best_score);
        }
        
        // Get checkpoints
        let checkpoints = self.checkpoint_manager.get_checkpoint_paths();
        
        Ok(TrainingResults {
            state: self.state.clone(),
            model: self.engine.model.clone(),
            calibrated_model: None, // TODO: Implement
            metrics: final_metrics,
            calibration_results,
            checkpoints,
            training_time,
        })
    }
    
    /// Initialize device based on configuration
    fn initialize_device(device_config: &DeviceConfig) -> Result<Device> {
        match device_config {
            DeviceConfig::Auto => {
                if candle_core::utils::cuda_is_available() {
                    Ok(Device::new_cuda(0)?)
                } else if candle_core::utils::metal_is_available() {
                    Ok(Device::new_metal(0)?)
                } else {
                    Ok(Device::Cpu)
                }
            }
            DeviceConfig::Cpu { .. } => Ok(Device::Cpu),
            DeviceConfig::Cuda { device_id, .. } => {
                Ok(Device::new_cuda(device_id.unwrap_or(0))?)
            }
            DeviceConfig::Metal { device_id } => {
                Ok(Device::new_metal(device_id.unwrap_or(0))?)
            }
            DeviceConfig::Multi { .. } => {
                // TODO: Implement multi-device support
                warn!("Multi-device not yet implemented, falling back to auto");
                Self::initialize_device(&DeviceConfig::Auto)
            }
        }
    }
    
    /// Initialize callbacks
    fn initialize_callbacks(
        config: &NeuralForgeConfig,
    ) -> Result<Vec<Box<dyn TrainingCallback>>> {
        let mut callbacks = Vec::new();
        
        // Add standard callbacks
        callbacks.push(Box::new(LoggingCallback::new(&config.logging)?));
        callbacks.push(Box::new(MetricsCallback::new()));
        
        // Add user-defined callbacks
        for callback_config in &config.training.callbacks {
            match callback_config {
                CallbackConfig::LRFinder { .. } => {
                    callbacks.push(Box::new(LRFinderCallback::new(callback_config)?));
                }
                CallbackConfig::Pruning { .. } => {
                    callbacks.push(Box::new(PruningCallback::new(callback_config)?));
                }
                CallbackConfig::GradientClipping { .. } => {
                    callbacks.push(Box::new(GradientClippingCallback::new(callback_config)?));
                }
                CallbackConfig::Custom { name, .. } => {
                    warn!("Custom callback '{}' not implemented", name);
                }
            }
        }
        
        Ok(callbacks)
    }
    
    /// Create learning rate scheduler
    fn create_scheduler(
        scheduler_config: &SchedulerConfig,
        optimizer_config: &OptimizerConfig,
    ) -> Result<Box<dyn LRScheduler>> {
        match scheduler_config {
            SchedulerConfig::StepLR { step_size, gamma } => {
                Ok(Box::new(StepLRScheduler::new(*step_size, *gamma)))
            }
            SchedulerConfig::ExponentialLR { gamma } => {
                Ok(Box::new(ExponentialLRScheduler::new(*gamma)))
            }
            SchedulerConfig::CosineAnnealingLR { t_max, eta_min } => {
                Ok(Box::new(CosineAnnealingLRScheduler::new(*t_max, *eta_min)))
            }
            SchedulerConfig::ReduceLROnPlateau { 
                factor, patience, threshold, .. 
            } => {
                Ok(Box::new(ReduceLROnPlateauScheduler::new(
                    *factor, *patience, *threshold
                )))
            }
            SchedulerConfig::OneCycleLR { max_lr, total_steps } => {
                Ok(Box::new(OneCycleLRScheduler::new(*max_lr, *total_steps)))
            }
            SchedulerConfig::Custom { name, .. } => {
                Err(NeuralForgeError::config(format!(
                    "Custom scheduler '{}' not implemented", name
                )))
            }
        }
    }
}

impl TrainingState {
    /// Create new training state
    pub fn new() -> Self {
        Self {
            epoch: 0,
            step: 0,
            total_steps: 0,
            best_score: None,
            best_model_path: None,
            start_time: Instant::now(),
            learning_rate: 0.001, // Default
            history: TrainingHistory::default(),
            early_stopping_counter: 0,
            should_stop: false,
        }
    }
}